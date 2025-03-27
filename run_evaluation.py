import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import s3fs
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient

from src.db_building import chroma_topk_to_df
from src.evaluation.basic_evaluation import answer_faq_by_bot, transform_answers_bot
from src.results_logging.mlflow_utils import retrieve_unique_collection_id
from src.utils.utils_vllm import get_model_from_env

# Logging configuration
# logger = logging.getLogger(__name__)


load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Parameters to control database building")
parser.add_argument(
    "--collection_name",
    type=str,
    default=None,
    help="Database collection name (for Qdrant or Chroma like database)",
)
parser.add_argument(
    "--mlflow_experiment_name_building",
    type=str,
    default="vector_database_building",
    help="Experiment name in mlflow",
)
parser.add_argument(
    "--mlflow_experiment_name_evaluation",
    type=str,
    default="vector_database_evaluation",
    help="Experiment name in mlflow",
)
parser.add_argument(
    "--top_k_statistics",
    type=int,
    default=10,
    help="Number of documents that should be given by retriever for evaluation",
)

args = parser.parse_args()


# CONFIGURATION ------------------------------------------

config_s3 = {"AWS_ENDPOINT_URL": os.getenv("AWS_ENDPOINT_URL", "https://minio.lab.sspcloud.fr")}

config_database_client = {
    "QDRANT_URL": os.getenv("QDRANT_URL", None),
    "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", None),
    "QDRANT_COLLECTION_NAME": args.collection_name,
}

config_mlflow = {
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", None),
    "MLFLOW_EXPERIMENT_NAME": args.mlflow_experiment_name_evaluation,
}

config_embedding_model = {
    # Assuming an OpenAI compatible client is used (VLLM, Ollama, etc.)
    "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE", os.getenv("URL_EMBEDDING_MODEL")),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "EMPTY"),
}


config = {**config_s3, **config_database_client, **config_mlflow, **config_embedding_model}


# PARAMETRES -------------------------------------

embedding_model = get_model_from_env("URL_EMBEDDING_MODEL")
s3_bucket = "projet-llm-insee-open-data"
faq_s3_path = "data/FAQ_site/faq.parquet"


# RETRIEVE DATABASE PARAMETERS

collection_name = args.collection_name
if args.collection_name is None:
    collection_name = retrieve_unique_collection_id(
        experiment_name=args.mlflow_experiment_name_building, embedding_model=embedding_model, config=config
    )


def run_evaluation() -> None:
    mlflow.set_tracking_uri(config.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(config.get("MLFLOW_EXPERIMENT_NAME"))
    filesystem = s3fs.S3FileSystem(endpoint_url=config.get("AWS_ENDPOINT_URL"))

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Logging the full configuration to mlflow

        # INPUT: FAQ THAT WILL BE USED FOR EVALUATION -----------------

        logger.info("Importing evaluation dataset")

        faq = pd.read_parquet(f"s3://{s3_bucket}/{faq_s3_path}", filesystem=filesystem)

        # Extract all URLs from the 'sources' column
        faq["urls"] = faq["sources"].str.findall(r"https?://www\.insee\.fr[^\s]*").apply(lambda s: ", ".join(s))

        # ------------------------
        # I - LOAD VECTOR DATABASE
        logger.info("Loading vector database")

        emb_model = OpenAIEmbeddings(
            model=embedding_model,
            base_url=config.get("OPENAI_API_BASE"),
            api_key=config.get("OPENAI_API_KEY"),
        )  # should we use src.utils.utils_vllm ?

        # Ensure correct database is used
        client = QdrantClient(
            url=config.get("QDRANT_URL"), api_key=config.get("QDRANT_API_KEY"), port="443", https="true"
        )
        logger.success("Connection to DB client successful")

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=emb_model,
            vector_name=embedding_model,
        )

        logger.success("Vectorstore initialization successful")

        # ------------------------
        # II - CREATING RETRIEVER

        logger.info(f"Creating retriever {80 * '='}")

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": args.top_k_statistics})

        # Log retriever
        retrieved_docs = retriever.invoke("Quels sont les chiffres du chômage en 2023 ?")
        result_retriever_raw = chroma_topk_to_df(retrieved_docs)
        mlflow.log_table(
            data=result_retriever_raw,
            artifact_file="retrieved_documents_retriever_raw.json",
        )

        # --------------------------------
        # III - RETRIEVER STATISTICS

        logger.info("Evaluating model performance against expectations")

        answers_bot = answer_faq_by_bot(retriever, faq)
        eval_reponses_bot, answers_bot_topk = transform_answers_bot(answers_bot, k=args.top_k_statistics)

        document_among_topk = answers_bot_topk["cumsum_url_expected"].max()
        document_is_top = answers_bot_topk["cumsum_url_expected"].min()

        mlflow.log_metric("document_is_first", 100 * document_is_top)
        mlflow.log_metric("document_among_topk", 100 * document_among_topk)
        mlflow.log_metrics(
            {
                f"document_in_top_{int(row['document_position'])}": 100 * row["cumsum_url_expected"]
                for _, row in answers_bot_topk.iterrows()
            }
        )
        mlflow.log_table(data=eval_reponses_bot, artifact_file="output/eval_reponses_bot.json")

        # --------------------------------
        # IV - OTHER USEFUL METADATA

        logger.info("Storing additional metadata")

        # Store FAQ
        mlflow_faq_raw = mlflow.data.from_pandas(faq, source=faq_s3_path, name="FAQ_data")
        mlflow.log_input(mlflow_faq_raw, context="faq-raw")
        mlflow.log_table(data=faq, artifact_file="faq_data.json")

        # Log environment necessary to reproduce the experiment
        current_dir = Path(".")
        FILES_TO_LOG = current_dir.glob("src/**/*.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            for file_path in FILES_TO_LOG:
                relative_path = file_path.relative_to(current_dir)
                destination_path = tmp_dir_path / relative_path.parent
                destination_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(file_path, destination_path)

            # Generate requirements.txt using pipreqs
            subprocess.run(["pipreqs", str(tmp_dir_path)], check=True)

            # Log all Python files to MLflow artifact
            mlflow.log_artifacts(tmp_dir, artifact_path="environment")

        # OLDIES ------------------------------
        # ------------------------
        # III - QUESTION VALIDATOR

        # logger.info("Testing the questions that are accepted/refused by our agent")

        # validator = build_chain_validator(evaluator_llm=llm, tokenizer=tokenizer)
        # validator_answers = evaluate_question_validator(validator=validator)
        # true_positive_validator = validator_answers.loc[validator_answers["real"], "real"].mean()
        # true_negative_validator = 1 - (validator_answers.loc[~validator_answers["real"], "real"].mean())
        # mlflow.log_metric("validator_true_positive", 100 * true_positive_validator)
        # mlflow.log_metric("validator_negative", 100 * true_negative_validator)

        # # ------------------------
        # # IV - RERANKER

        # if config.reranking_method is not None:
        #     logger.info(f"Applying reranking {80 * '='}")
        #     logger.info(f"Selected method: {config.reranking_method}")

        #     # Define a langchain prompt template
        #     RAG_PROMPT_TEMPLATE_RERANKER = tokenizer.apply_chat_template(
        #         get_chatbot_template(), tokenize=False, add_generation_prompt=True
        #     )
        #     prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE_RERANKER)

        #     mlflow.log_dict(get_chatbot_template(config)[0], "chatbot_template.json")

        #     chain = build_chain(
        #         retriever=retriever,
        #         prompt=prompt,
        #         llm=llm,
        #         reranker=config.reranking_method,
        #     )
        # else:
        #     logger.info(f"Skipping reranking since value is None {80 * '='}")

        # # ------------------------
        # # V - EVALUATION

        # logger.info(f"Evaluating model performance against expectations {80 * '='}")

        # if config.reranking_method is None:
        #     answers_bot = answer_faq_by_bot(retriever, faq)
        #     eval_reponses_bot, answers_bot_topk = transform_answers_bot(answers_bot, k=config.topk_stats)
        # else:
        #     answers_bot_before_reranker = answer_faq_by_bot(retriever, faq)
        #     eval_reponses_bot_before_reranker, answers_bot_topk_before_reranker = transform_answers_bot(
        #         answers_bot_before_reranker, k=5
        #     )
        #     answers_bot_after_reranker = answer_faq_by_bot(chain, faq)
        #     eval_reponses_bot_after_reranker, answers_bot_topk_after_reranker = transform_answers_bot(
        #         answers_bot_after_reranker, k=5
        #     )
        #     eval_reponses_bot = compare_performance_reranking(
        #         eval_reponses_bot_after_reranker, eval_reponses_bot_before_reranker
        #     )
        #     answers_bot_topk = answers_bot_topk_after_reranker

        # # Compute model performance at the end of the pipeline
        # document_among_topk = answers_bot_topk["cumsum_url_expected"].max()
        # document_is_top = answers_bot_topk["cumsum_url_expected"].min()
        # # Also compute model performance before reranking when relevant
        # if config.reranking_method is not None:
        #     document_among_topk_before_reranker = answers_bot_topk_before_reranker["cumsum_url_expected"].max()
        #     document_is_top_before_reranker = answers_bot_topk_before_reranker["cumsum_url_expected"].min()

        # # Store FAQ
        # mlflow_faq_raw = mlflow.data.from_pandas(faq, source=config.faq_s3_uri, name="FAQ_data")
        # mlflow.log_input(mlflow_faq_raw, context="faq-raw")
        # mlflow.log_table(data=faq, artifact_file="faq_data.json")

        # # Check if document expected is in topk answers =========================
        # mlflow.log_metric("document_is_first", 100 * document_is_top)
        # mlflow.log_metric("document_among_topk", 100 * document_among_topk)
        # mlflow.log_metrics(
        #     {
        #         f"document_in_top_{int(row['document_position'])}": 100 * row["cumsum_url_expected"]
        #         for _, row in answers_bot_topk.iterrows()
        #     }
        # )
        # mlflow.log_table(data=eval_reponses_bot, artifact_file="output/eval_reponses_bot.json")

        # # If we used reranking, we also store performance before reranking
        # if config.reranking_method is not None:
        #     mlflow.log_metric("document_is_first_before_reranker", 100 * document_is_top_before_reranker)
        #     mlflow.log_metric("document_among_topk_before_reranker", 100 * document_among_topk_before_reranker)
        #     mlflow.log_metrics(
        #         {
        #             f"document_in_top_{int(row['document_position'])}_before_reranker": 100 * row["cumsum_url_expected"]
        #             for _, row in answers_bot_topk_before_reranker.iterrows()
        #         }
        #     )

        # # Log environment necessary to reproduce the experiment
        # current_dir = Path(".")
        # FILES_TO_LOG = list(current_dir.glob("src/db_building/*.py")) + list(current_dir.glob("src/config/*.py"))

        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     tmp_dir_path = Path(tmp_dir)

        #     for file_path in FILES_TO_LOG:
        #         relative_path = file_path.relative_to(current_dir)
        #         destination_path = tmp_dir_path / relative_path.parent
        #         destination_path.mkdir(parents=True, exist_ok=True)
        #         shutil.copy(file_path, destination_path)

        #     # Generate requirements.txt using pipreqs
        #     subprocess.run(["pipreqs", str(tmp_dir_path)], check=True)

        #     # Log all Python files to MLflow artifact
        #     mlflow.log_artifacts(tmp_dir, artifact_path="environment")


run_evaluation()

# if __name__ == "__main__":
#     argparser = llm_argparser()
#     load_config(argparser)
#     assert DefaultFullConfig().mlflow_tracking_uri is not None, "Please set the mlflow_tracking_uri parameter."
#     assert os.environ.get("HF_TOKEN"), "Please set the HF_TOKEN environment variable."
#     filesystem = s3fs.S3FileSystem(endpoint_url=DefaultFullConfig().s3_endpoint_url)
#     run_evaluation(filesystem)
