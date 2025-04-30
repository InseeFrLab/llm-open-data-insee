import argparse
import os

import mlflow
import pandas as pd
import s3fs
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from loguru import logger
import openai
from openai import OpenAI

from src.config import set_config

from src.data.caching import chunk_documents_or_load_from_cache, parse_documents_or_load_from_cache
from src.vectordatabase.chroma import chroma_vectorstore_as_retriever
from src.vectordatabase.client import create_client_and_collection, get_number_docs_collection
from src.vectordatabase.embed_by_piece import chunk_documents_and_store
from src.vectordatabase.qdrant import qdrant_vectorstore_as_retriever

from src.evaluation.basic_evaluation import compare_retriever_with_expected_docs, transform_answers_bot
from src.evaluation.hallucination import check_hallucination_rate

from src.results_logging.mlflow_utils import mlflow_log_source_files
from src.utils.utils_vllm import get_models_from_env

load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Parameters to control database building")
parser.add_argument(
    "--collection_name",
    type=str,
    default="dirag_experimentation",
    help="Database collection name (for Qdrant or Chroma like database)",
)
parser.add_argument(
    "--max_pages",
    type=int,
    default=None,
    help="Max number of pages that should be considered",
)
parser.add_argument(
    "--mlflow_experiment_name",
    type=str,
    default="vector_database_building",
    help="Experiment name in mlflow",
)
parser.add_argument(
    "--mlflow_run_id",
    type=str,
    default=None,
    help="MLFlow run id we should restart from",
)
parser.add_argument(
    "--chunking_strategy",
    type=str,
    default="None",
    choices=["None", "Recursive", "recursive", "Character", "character"],
    help="Chunking strategy for documents"
    "If None (default), corpus is left asis. "
    "If recursive, use Langchain's CharacterTextSplitter",
)
parser.add_argument(
    "--max_document_size",
    type=int,
    default=1200,
    help="Threshold size for documents. "
    "If None (default), corpus is left asis. "
    "If value provided, passed to Langchain's CharacterTextSplitter is applied",
)
parser.add_argument(
    "--chunk_overlap",
    type=int,
    default=100,
    help="Chunk overlap when documents are split. "
    "If value provided, passed to Langchain's CharacterTextSplitter is applied",
)
parser.add_argument(
    "--chunking_strategy_number_of_splits",
    type=int,
    default=100,
    help="How many splits of the dataset should we use to construct database",
)

parser.add_argument(
    "--dataset",
    choices=["dirag", "complete"],
    default="complete",
    help="""
        Choose the dataset type: 'dirag' for restricted DIRAG data,
        'complete' for the full web4g dataset (default: 'complete'),
    """,
)
parser.add_argument(
    "--top_k_statistics",
    type=int,
    default=10,
    help="Number of documents that should be given by retriever for evaluation",
)
parser.add_argument(
    "--database_engine",
    type=str,
    default="qdrant",
    help="Vector database engine",
)
parser.add_argument("--verbose", action="store_true", help="Enable verbose output (default: False)")
parser.add_argument(
    "--invalidate_cache",
    action="store_true",
    help="Reconstruct cached documents (default: False) [not implemented right now]",
)


# Example usage:
# python run_build_dataset.py max_pages 10 --dataset dirag
# python run_build_dataset.py max_pages 10

args = parser.parse_args()

# Logging configuration
# logger = logging.getLogger(__name__)

# CONFIGURATION ------------------------------------------

engine = args.database_engine

config = set_config(
    use_vault=True,
    components=["s3", "mlflow", "database", "model"],
    mlflow_experiment_name=args.mlflow_experiment_name,
    database_manager=engine,
    models_location={
        "url_embedding_model": "ENV_URL_EMBEDDING_MODEL",
        "url_generative_model": "ENV_URL_GENERATIVE_MODEL",
    },
    override={f"{engine}_COLLECTION_NAME": args.collection_name},
    verbose=args.verbose,
)


filtered_config = {k: v for k, v in config.items() if not any(s in k.lower() for s in ["key", "token", "secret"])}
filtered_config.pop(f"{engine.upper()}_COLLECTION_NAME", None)

embedding_model = get_models_from_env(url_embedding="URL_EMBEDDING_MODEL", config_dict=config).get("embedding")
generative_model = get_models_from_env(url_embedding="URL_GENERATIVE_MODEL", config_dict=config).get("embedding")

url_database_client = config.get(f"{engine.upper()}_URL")
api_key_database_client = config.get(f"{engine.upper()}_API_KEY")


logger.debug(f"Using {embedding_model} for database retrieval")
logger.debug(f"Setting {url_database_client} as vector database endpoint")


# PARAMETERS ------------------------------------------

S3_PATH = "s3://projet-llm-insee-open-data/data/raw_data/applishare_solr_joined.parquet"
collection_name = args.collection_name
s3_bucket = "projet-llm-insee-open-data"

max_document_size = args.max_document_size
chunk_overlap = args.chunk_overlap

filesystem = s3fs.S3FileSystem(endpoint_url=config.get("endpoint_url"))


corpus_constructor_args = {
    "field": args.dataset,
    "web4g_path_uri": S3_PATH,
    "fs": filesystem,
    "search_cols": ["titre", "libelleAffichageGeo", "xml_intertitre"],
}

chunking_args = {"strategy": args.chunking_strategy, "chunk_size": max_document_size, "chunk_overlap": chunk_overlap}

parameters_database_construction = {
    "embedding_model": embedding_model,
    f"{engine.upper()}_URL_API": url_database_client,
    "chunking_strategy": args.chunking_strategy,
    "max_document_size": max_document_size,
    "chunk_overlap": chunk_overlap,
    "engine": engine,
}


CACHE_DIR = f"s3://projet-llm-insee-open-data/data/intermediate/dataset={args.dataset}"
path_cached_parsed_documents = f"{CACHE_DIR}/web4g_parsed/parsed_data.parquet"
path_cached_chunked_documents = (
    f"{CACHE_DIR}/web4g_chunked/"
    f"strategy={args.chunking_strategy}/chunk_size={max_document_size}/"
    f"chunk_overlap={chunk_overlap}/data.jsonl"
)

if args.dataset == "dirag":
    logger.warning("Restricting publications to DIRAG related content")


# LOADING EMBEDDING AND COMPLETION MODELS ---------------------------------

emb_model = OpenAIEmbeddings(
    model=embedding_model,
    openai_api_base=config.get("OPENAI_API_BASE_EMBEDDING"),
    openai_api_key=config.get("OPENAI_API_KEY_EMBEDDING"),
    tiktoken_enabled=False,
)

model_max_len = len(emb_model.embed_query("retrieving hidden_size"))

chat_client = OpenAI(
    base_url=config.get("OPENAI_API_BASE_GENERATIVE"),
    api_key=config.get("OPENAI_API_KEY_GENERATIVE"),
)

# LOADING PROMPT -----------------------------------------------

with open("./prompt/question.md", encoding="utf-8") as f:
    question_prompt = f.read()

with open("./prompt/system.md", encoding="utf-8") as f:
    system_prompt = f.read()


# MAIN PIPELINE --------------------------------------

logger.info("Connecting to vector database")


def run_build_database() -> None:
    mlflow.set_tracking_uri(config.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(config.get("MLFLOW_EXPERIMENT_NAME"))
    mlflow.langchain.autolog()

    with mlflow.start_run(args.mlflow_run_id) as run:
        # new run or starting from an existing one
        run_id = run.info.run_id

        mlflow.log_params(
            {
                **filtered_config,
                **parameters_database_construction,
                **{"dataset": args.dataset, "max_pages": args.max_pages, "max_document_size": max_document_size},
            }
        )

        # PARSING CORPUS -----------------------------------

        logger.debug(f"Importing and parsing raw data {30 * '-'}")

        data = parse_documents_or_load_from_cache(
            path_for_cache=path_cached_parsed_documents,
            load_from_cache=True,
            max_pages=args.max_pages,
            filesystem=filesystem,
            corpus_constructor_args=corpus_constructor_args,
        )

        data["mlflow_run_id"] = run_id  # Add mlflow run id in metadata

        # CHUNKING DOCUMENTS --------------------------

        loader = DataFrameLoader(data, page_content_column="content")
        documents = loader.load()

        logger.info(f"Before chunking, we have {len(documents)} pages")

        documents = chunk_documents_or_load_from_cache(
            documents_before_chunking=documents,
            path_for_cache=path_cached_chunked_documents,
            max_pages=args.max_pages,
            load_from_cache=True,
            filesystem=filesystem,
            chunking_args=chunking_args,
        )

        logger.info(f"After chunking, we have {len(documents)} documents")

        # CREATE DATABASE COLLECTION -----------------------

        unique_collection_name = f"{collection_name}_{run_id}"

        client = create_client_and_collection(
            url=url_database_client,
            api_key=api_key_database_client,
            collection_name=unique_collection_name,
            model_max_len=model_max_len,
            engine=engine,
            vector_name=embedding_model,
        )

        # EMBEDDING DOCUMENTS IN VECTOR DATABASE -----------------------

        logger.info("Putting documents in vector database")

        n_documents = get_number_docs_collection(client=client, collection_name=unique_collection_name, engine=engine)

        if n_documents > 0 and args.invalidate_cache is False:
            message = "Skipping documents embedding sincecollection has {n_documents} documents"
            logger.info(message)
        else:
            chunk_documents_and_store(
                documents,
                emb_model,
                collection_name=unique_collection_name,
                url=url_database_client,
                api_key=api_key_database_client,
                engine=engine,
                client=client,
                number_chunks=args.chunking_strategy_number_of_splits,
            )

        mlflow.log_params({"COLLECTION_UNIQUE": unique_collection_name, "embedding_model": embedding_model})

        mlflow.log_metric("n_documents", n_documents)

        # TURNING DATABASE INTO RETRIEVER ----------------------

        constructor_retriever = qdrant_vectorstore_as_retriever
        if engine == "chroma":
            constructor_retriever = chroma_vectorstore_as_retriever

        retriever = constructor_retriever(
            client=client,
            collection_name=unique_collection_name,
            embedding_function=emb_model,
            vector_name=emb_model.model,
            number_retrieved_docs=10,
        )

        # PART II : LOGGING PROMPTS AND MODELS ---------------------------------

        mlflow.register_prompt(
            name="system-prompt-generative-model",
            template=system_prompt,
        )

        mlflow.register_prompt(
            name="question-prompt-generative-model",
            template=question_prompt,
        )


        # PART III : RETRIEVER EVALUATION ---------------------------------

        logger.info("Evaluating model performance against expectations")

        annotations = pd.read_csv(os.environ["ANNOTATIONS_LOCATION"])
        annotations = annotations.rename(
            {"Lien(s) attendu(s) vers site insee (si plusieurs: séparer avec des  ;)": "url"}, axis="columns"
        )
        annotations["url"] = annotations["url"].str.replace(" ", "")
        annotations["url"] = annotations["url"].str.split(";")


        # Starting tracing OpenAI (not before because vector DB embedding is too much to trace)
        mlflow.openai.autolog()

        # TODO: Vérifier si y a pas moyen de faire en batch
        answers_retriever, answers_generative, answers_generative_no_context = compare_retriever_with_expected_docs(
            retriever=retriever,
            ground_truth_df=annotations,
            question_col="Question",
            ground_truth_col="url",
            with_generation=True,
            chat_client=chat_client,
            chat_client_options={"model": generative_model},
        )

        answers_pipeline = pd.concat([
            annotations,
            pd.DataFrame({"answer_rag": answers_generative, "answer_no_rag": answers_generative_no_context})
            ],
            axis=1
        )

        url_suggested = answers_retriever.groupby('question').agg({
            'url': lambda x: '; '.join(x.dropna().astype(str))
        }).reset_index()

        answers_pipeline = answers_pipeline.merge(
            url_suggested,
            left_on='Question',
            right_on='question',
            how='left'
        )

        mlflow.log_table(data=answers_pipeline, artifact_file="output/qabot_eval_results.json")


        mlflow.log_metrics(
            {
                "hallucination_rate_rag": check_hallucination_rate(answers_generative),
                "hallucination_rate_no_rag": check_hallucination_rate(answers_generative_no_context)
            }
        )

        eval_reponses_bot, answers_bot_topk = transform_answers_bot(answers_retriever, k=args.top_k_statistics)

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
        mlflow.log_table(data=eval_reponses_bot, artifact_file="evaluation/annotations_avec_reponses.json")


        # LOGGING OTHER USEFUL THINGS --------------------------

        logger.info("Storing additional metadata")

        # Store FAQ
        annotations_raw = mlflow.data.from_pandas(
            annotations, source=os.environ["ANNOTATIONS_LOCATION"], name="Annotations"
        )
        mlflow.log_input(annotations_raw, context="annotations")
        mlflow.log_table(data=annotations, artifact_file="evaluation/annotations.json")

        # Log raw dataset built from web4g
        mlflow_data_raw = mlflow.data.from_pandas(
            data.head(10),
            source=S3_PATH,
            name="web4g_data",
        )
        mlflow.log_input(mlflow_data_raw, context="pre-embedding")
        mlflow.log_table(data=data.head(10), artifact_file="web4g_data.json")

        # Log src/*.py and run_build_database.py
        mlflow_log_source_files()

        logger.info("Program ended with success.")


if __name__ == "__main__":
    run_build_database()
