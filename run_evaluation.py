import argparse
import ast
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import s3fs
from langchain_core.prompts import PromptTemplate

from src.chain_building import build_chain_validator
from src.chain_building.build_chain import build_chain
from src.config import CHATBOT_TEMPLATE, CHROMA_DB_LOCAL_DIRECTORY, RAG_PROMPT_TEMPLATE
from src.db_building import chroma_topk_to_df, load_retriever, load_vector_database
from src.evaluation import answer_faq_by_bot, compare_performance_reranking, evaluate_question_validator, transform_answers_bot
from src.model_building import build_llm_model

# Logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.DEBUG,
)


# Command-line arguments
def str_to_list(arg):
    # Convert the argument string to a list
    return ast.literal_eval(arg)


# PARSER FOR USED LEVEL ARGUMENTS --------------------------------

parser = argparse.ArgumentParser(description="LLM building parameters")
parser.add_argument(
    "--experiment_name",
    type=str,
    default="default",
    help="""
    Name of the experiment.
    """,
)

## Optional database arguments ----
parser.add_argument(
    "--data_raw_s3_path",
    type=str,
    help="""
    Path to the raw data.
    """,
)
parser.add_argument(
    "--collection_name",
    type=str,
    help="""
    Collection name.
    """,
)
parser.add_argument(
    "--markdown_split",
    action=argparse.BooleanOptionalAction,
    help="""
    Should we use a markdown split ?
    --markdown_split yields True and --no-markdown_split yields False
    """,
)
parser.add_argument(
    "--use_tokenizer_to_chunk",
    action=argparse.BooleanOptionalAction,
    help="""
    Should we use the tokenizer of the embedding model to chunk ?
    --use_tokenizer_to_chunk yields True and --no-use_tokenizer_to_chunk yields False
    """,
)
parser.add_argument(
    "--separators",
    type=str_to_list,
    help="List separators to split the text",
)
parser.add_argument(
    "--embedding_model",
    type=str,
    help="""
    Embedding model.
    Should be a huggingface model.
    """,
)
parser.add_argument(
    "--chunk_size",
    type=str,
    help="""
    Chunk size
    """,
)
parser.add_argument(
    "--chunk_overlap",
    type=str,
    help="""
    Chunk overlap
    """,
)
# Either we define arguments or we give mlflow run id
parser.add_argument(
    "--database_run_id",
    type=str,
    default=None,
    help="""
    Mlflow run id of the database building.
    """,
)

## LLM specific arguments ----
parser.add_argument(
    "--llm_model",
    type=str,
    default=os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"),
    help="""
    LLM used to generate chat.
    Should be a huggingface model.
    Defaults to mistralai/Mistral-7B-Instruct-v0.2
    """,
)
parser.add_argument(
    "--quantization",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="""
    Should we use a quantized version of "model" argument ?
    --quantization yields True and --no-quantization yields False
    """,
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=2000,
    help="""
    The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    See https://huggingface.co/docs/transformers/main_classes/text_generation
    """,
)
parser.add_argument(
    "--model_temperature",
    type=int,
    default=0.2,
    help="""
    The value used to modulate the next token probabilities.
    See https://huggingface.co/docs/transformers/main_classes/text_generation
    """,
)
parser.add_argument(
    "--return_full_text",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="""
    Should we return the full text ?
    --return_full_text yields True and --no-return_full_text yields False
    Default to True
    """,
)
parser.add_argument(
    "--do_sample",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="""
    if set to True , this parameter enables decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling
    and Top-p sampling. All these strategies select the next token from the probability distribution over the entire vocabulary
    with various strategy-specific adjustments.
    --do_sample yields True and --no-do_sample yields False
    Default to True
    """,
)

parser.add_argument(
    "--reranking_method",
    type=str,
    default=None,
    help="""
    Reranking document relevancy after retrieval phase.
    Defaults to None (no reranking)
    """,
)
parser.add_argument(
    "--topk_stats",
    type=int,
    default=5,
    help="""
    Number of links considered to evaluate retriever quality.
    """,
)


args = parser.parse_args()


def run_evaluation(
    experiment_name: str,
    **kwargs,
):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)

    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": f"""https://{os.environ["AWS_S3_ENDPOINT"]}"""})

    # INPUT: FAQ THAT WILL BE USED FOR EVALUATION -----------------
    bucket = "projet-llm-insee-open-data"
    path = "data/FAQ_site/faq.parquet"
    faq = pd.read_parquet(f"{bucket}/{path}", filesystem=fs)
    # Extract all URLs from the 'sources' column
    faq["urls"] = faq["sources"].str.findall(r"https?://www\.insee\.fr[^\s]*").apply(lambda s: ", ".join(s))

    # Log parameters
    for arg_name, arg_value in locals().items():
        if arg_name == "kwargs":
            for key, value in arg_value.items():
                mlflow.log_param(key, value)
        else:
            mlflow.log_param(arg_name, arg_value)

        # ------------------------
        # I - LOAD VECTOR DATABASE

        db = load_vector_database(filesystem=fs, **kwargs)

        # ------------------------
        # II - CREATING RETRIEVER

        logging.info(f"Training retriever {80*'='}")

        mlflow.log_text(RAG_PROMPT_TEMPLATE, "rag_prompt.md")

        llm, tokenizer = build_llm_model(
            model_name=kwargs.get("llm_model"),
            quantization_config=kwargs.get("quantization"),
            config=True,
            token=os.getenv("HF_TOKEN"),
            streaming=False,
            generation_args=kwargs,
        )

        logging.info("Logging an example of tokenized text")
        query = "Quels sont les chiffres du chômages en 2023 ?"
        mlflow.log_text(
            f"{query} \n ---------> \n {', '.join(tokenizer.tokenize(query))}",
            "example_tokenizer.json",
        )

        retriever, vectorstore = load_retriever(
            emb_model_name=kwargs.get("embedding_model"),
            vectorstore=db,
            persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
            retriever_params={"search_type": "similarity", "search_kwargs": {"k": 30}},
        )

        # Log retriever
        retrieved_docs = retriever.invoke("Quels sont les chiffres du chômage en 2023 ?")
        result_retriever_raw = chroma_topk_to_df(retrieved_docs)
        mlflow.log_table(
            data=result_retriever_raw,
            artifact_file="retrieved_documents_retriever_raw.json",
        )

        # ------------------------
        # III - QUESTION VALIDATOR

        logging.info("Testing the questions that are accepted/refused by our agent")

        validator = build_chain_validator(evaluator_llm=llm, tokenizer=tokenizer)
        validator_answers = evaluate_question_validator(validator=validator)
        true_positive_validator = validator_answers.loc[validator_answers["real"], "real"].mean()
        true_negative_validator = 1 - (validator_answers.loc[~validator_answers["real"], "real"].mean())
        mlflow.log_metric("validator_true_positive", 100 * true_positive_validator)
        mlflow.log_metric("validator_negative", 100 * true_negative_validator)

        # ------------------------
        # IV - RERANKER

        reranking_method = kwargs.get("reranking_method")

        if reranking_method is not None:
            logging.info(f"Applying reranking {80*'='}")
            logging.info(f"Selected method: {reranking_method}")
        else:
            logging.info(f"Skipping reranking since value is None {80*'='}")

        if reranking_method is not None:
            # Define a langchain prompt template
            RAG_PROMPT_TEMPLATE_RERANKER = tokenizer.apply_chat_template(CHATBOT_TEMPLATE, tokenize=False, add_generation_prompt=True)
            prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE_RERANKER)

            mlflow.log_dict(CHATBOT_TEMPLATE, "chatbot_template.json")

            chain = build_chain(
                retriever=retriever,
                prompt=prompt,
                llm=llm,
                bool_log=False,
                reranker=reranking_method,
            )

        # ------------------------
        # V - EVALUATION

        logging.info(f"Evaluating model performance against expectations {80*'='}")

        if reranking_method is None:
            answers_bot = answer_faq_by_bot(retriever, faq)
            eval_reponses_bot, answers_bot_topk = transform_answers_bot(answers_bot, k=kwargs.get("topk_stats"))
        else:
            answers_bot_before_reranker = answer_faq_by_bot(retriever, faq)
            eval_reponses_bot_before_reranker, answers_bot_topk_before_reranker = transform_answers_bot(answers_bot_before_reranker, k=5)
            answers_bot_after_reranker = answer_faq_by_bot(chain, faq)
            eval_reponses_bot_after_reranker, answers_bot_topk_after_reranker = transform_answers_bot(answers_bot_after_reranker, k=5)
            eval_reponses_bot = compare_performance_reranking(eval_reponses_bot_after_reranker, eval_reponses_bot_before_reranker)
            answers_bot_topk = answers_bot_topk_after_reranker

        # Compute model performance at the end of the pipeline
        document_among_topk = answers_bot_topk["cumsum_url_expected"].max()
        document_is_top = answers_bot_topk["cumsum_url_expected"].min()
        # Also compute model performance before reranking when relevant
        if reranking_method is not None:
            document_among_topk_before_reranker = answers_bot_topk_before_reranker["cumsum_url_expected"].max()
            document_is_top_before_reranker = answers_bot_topk_before_reranker["cumsum_url_expected"].min()

        # Store FAQ
        mlflow_faq_raw = mlflow.data.from_pandas(
            faq,
            source=f"s3://{bucket}/{path}",
            name="FAQ_data",
        )
        mlflow.log_input(mlflow_faq_raw, context="faq-raw")
        mlflow.log_table(data=faq, artifact_file="faq_data.json")

        # Check if document expected is in topk answers =========================
        mlflow.log_metric("document_is_first", 100 * document_is_top)
        mlflow.log_metric("document_among_topk", 100 * document_among_topk)
        mlflow.log_metrics(
            {f'document_in_top_{int(row["document_position"])}': 100 * row["cumsum_url_expected"] for _, row in answers_bot_topk.iterrows()}
        )
        mlflow.log_table(data=eval_reponses_bot, artifact_file="output/eval_reponses_bot.json")

        # If we used reranking, we also store performance before reranking
        if reranking_method is not None:
            mlflow.log_metric("document_is_first_before_reranker", 100 * document_is_top_before_reranker)
            mlflow.log_metric("document_among_topk_before_reranker", 100 * document_among_topk_before_reranker)
            mlflow.log_metrics(
                {
                    f'document_in_top_{int(row["document_position"])}_before_reranker': 100 * row["cumsum_url_expected"]
                    for _, row in answers_bot_topk_before_reranker.iterrows()
                }
            )

        # Log environment necessary to reproduce the experiment
        current_dir = Path(".")
        FILES_TO_LOG = list(current_dir.glob("src/db_building/*.py")) + list(current_dir.glob("src/config/*.py"))

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


if __name__ == "__main__":
    assert "MLFLOW_TRACKING_URI" in os.environ, "Please set the MLFLOW_TRACKING_URI environment variable."

    assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable."

    args = parser.parse_args()

    run_evaluation(
        **vars(args),
    )
