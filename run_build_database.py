import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path, PosixPath

import mlflow
import pandas as pd
import requests
import s3fs
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from src.config import set_config
from src.data.corpus import constructor_corpus
from src.vectordatabase.output_parsing import langchain_documents_to_df
from src.vectordatabase.document_chunker import chunk_documents, parse_documents
from src.vectordatabase.embed_by_piece import chunk_documents_and_store
from src.vectordatabase.qdrant import (
    create_client_and_collection_qdrant
)
from src.evaluation.basic_evaluation import answer_faq_by_bot, transform_answers_bot
from src.utils.prompt import similarity_search_instructions
from src.utils.utils_vllm import get_models_from_env, get_model_max_len

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
    "--chunking_strategy",
    type=str,
    default="None",
    choices=["None", "Recursive", "recursive"],
    help="Chunking strategy for documents"
    "If None (default), corpus is left asis. "
    "If recursive, use Langchain's CharacterTextSplitter",
)
parser.add_argument(
    "--max_document_size",
    type=int,
    default=2000,
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
    "--dataset",
    choices=["dirag", "complete"],
    default="complete",
    help="Choose the dataset type: 'dirag' for restricted DIRAG data, 'complete' for the full web4g dataset (default: 'complete').",
)
parser.add_argument("--verbose", action="store_true", help="Enable verbose output (default: False)")
parser.add_argument(
    "--log_database_snapshot", action="store_true", help="Should we log database snapshot ? (default: False)"
)
parser.add_argument(
    "--top_k_statistics",
    type=int,
    default=10,
    help="Number of documents that should be given by retriever for evaluation",
)
# Example usage:
# python run_build_dataset.py max_pages 10 --dataset dirag
# python run_build_dataset.py max_pages 10

args = parser.parse_args()

# Logging configuration
# logger = logging.getLogger(__name__)

# CONFIGURATION ------------------------------------------

config = set_config(
    use_vault=True,
    components=["s3", "mlflow", "database", "model"],
    mlflow_experiment_name=args.mlflow_experiment_name,
    models_location={
        "url_embedding_model": "ENV_URL_EMBEDDING_MODEL",
        "url_generative_model": "ENV_URL_GENERATIVE_MODEL",
    },
    override={"QDRANT_COLLECTION_NAME": args.collection_name},
    verbose=args.verbose,
)


# PARAMETERS ------------------------------------------

S3_PATH = "s3://projet-llm-insee-open-data/data/raw_data/applishare_solr_joined.parquet"
collection_name = args.collection_name
s3_bucket = "projet-llm-insee-open-data"
faq_s3_path = "data/FAQ_site/faq.parquet"

url_database_client = config.get("QDRANT_URL")
api_key_database_client = config.get("QDRANT_API_KEY")
max_document_size = args.max_document_size
chunk_overlap = args.chunk_overlap
embedding_model = get_models_from_env(url_embedding="OPENAI_API_BASE_EMBEDDING", config_dict=config).get("embedding")

parameters_database_construction = {
    "embedding_model": embedding_model,
    "QDRANT_URL_API": url_database_client,
    "chunking_strategy": args.chunking_strategy,
    "max_document_size": max_document_size,
    "chunk_overlap": chunk_overlap,
}

logger.debug(f"Using {embedding_model} for database retrieval")
logger.debug(f"Setting {url_database_client} as vector database endpoint")

if args.dataset == "dirag":
    logger.warning("Restricting publications to DIRAG related content")


def run_build_database() -> None:
    mlflow.set_tracking_uri(config.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(config.get("MLFLOW_EXPERIMENT_NAME"))

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        filesystem = s3fs.S3FileSystem(endpoint_url=config.get("endpoint_url"))

        filtered_config = {
            k: v for k, v in config.items() if not any(s in k.lower() for s in ["key", "token", "secret"])
        }

        mlflow.log_params(filtered_config)
        mlflow.log_params(parameters_database_construction)
        mlflow.log_params({"dataset": args.dataset})

        # LOAD AND PROCESS CORPUS -----------------------------------
        logger.debug(f"Importing raw data {30 * '-'}")

        data = constructor_corpus(
            field=args.dataset,
            web4g_path_uri=S3_PATH,
            fs=filesystem,
            search_cols=["titre", "libelleAffichageGeo", "xml_intertitre"],
        )

        data["mlflow_run_id"] = run_id  # Add mlflow run id in metadata

        if args.max_pages is not None:
            logger.debug(f"Limiting database to {args.max_pages} pages")
            data = data.head(args.max_pages)

        mlflow.log_param("max_pages", args.max_pages)
        mlflow.log_param("max_document_size", max_document_size)

        logger.info("Starting to parse XMLs")

        documents = parse_documents(data=data, engine_output="langchain")

        logger.success("XMLs have been parsed")

        # SPLITTING STRATEGY -------------------------------

        if args.chunking_strategy != "None":
            documents = chunk_documents(documents, **{"chunk_size": max_document_size, "chunk_overlap": chunk_overlap})

        # CREATE DATABASE COLLECTION -----------------------

        logger.info("Connecting to vector database")

        model_max_len = get_model_max_len(model_id=embedding_model)
        unique_collection_name = f"{collection_name}_{run_id}"

        client = create_client_and_collection_qdrant(
            url=url_database_client,
            api_key=api_key_database_client,
            collection_name=unique_collection_name,
            model_max_len=model_max_len,
        )

        emb_model = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=config.get("OPENAI_API_BASE_EMBEDDING"),
            openai_api_key=config.get("OPENAI_API_KEY_EMBEDDING"),
        )

        # EMBEDDING DOCUMENTS IN VECTOR DATABASE -----------------------
        logger.info("Putting documents in vector database")

        chunk_documents_and_store(
            documents,
            emb_model,
            collection_name=unique_collection_name,
            url=url_database_client,
            api_key=api_key_database_client,
        )

        # SETTING ALIAS -----------------------

        client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(collection_name=unique_collection_name, alias_name=collection_name)
                )
            ]
        )

        mlflow.log_params(
            {
                "QDRANT_COLLECTION_UNIQUE": unique_collection_name,
                "QDRANT_COLLECTION_ALIAS": collection_name,
            }
        )

        # LOGGING DATABASE STATISTICS --------------------------

        collection_info = client.get_collection(collection_name=unique_collection_name)
        embedding_size = collection_info.config.params.vectors.get(embedding_model).size

        n_documents = collection_info.points_count

        mlflow.log_params(
            {"embedding_size": embedding_size, "n_documents": n_documents, "embedding_model": embedding_model}
        )

        # CREATING SNAPSHOT FOR LOGGING -------------------
        if args.log_database_snapshot is True:
            logger.info("Logging database snapshot")

            snapshot = client.create_snapshot(collection_name=unique_collection_name)

            url_snapshot = f"{url_database_client}/collections/{unique_collection_name}/snapshots/{snapshot.name}"

            # Intermediate save snapshot in local for logging in MLFlow
            response = requests.get(url_snapshot, headers={"api-key": api_key_database_client}, timeout=60 * 10)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".snapshot") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name  # Store temp file path

            mlflow.log_artifact(local_path=temp_file_path)

            logger.success("Database building successful")

        # PART II : EVALUATION ---------------------------------

        logger.info("Importing evaluation dataset")

        faq = pd.read_parquet(f"s3://{s3_bucket}/{faq_s3_path}", filesystem=filesystem)
        # Extract all URLs from the 'sources' column
        faq["urls"] = faq["sources"].str.findall(r"https?://www\.insee\.fr[^\s]*").apply(lambda s: ", ".join(s))

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        # Log a result of a similarity search
        query = f"{similarity_search_instructions}\nQuery: Quels sont les chiffres du chômage en 2023 ?"
        mlflow.log_param("prompt_retriever", similarity_search_instructions)
        retrieved_docs = retriever.invoke(query)
        result_retriever_raw = langchain_documents_to_df(retrieved_docs)

        mlflow.log_table(data=result_retriever_raw, artifact_file="retrieved_documents.json")
        mlflow.log_param("question_asked", query)

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

        # LOGGING OTHER USEFUL THINGS --------------------------

        logger.info("Storing additional metadata")

        # Store FAQ
        mlflow_faq_raw = mlflow.data.from_pandas(faq, source=faq_s3_path, name="FAQ_data")
        mlflow.log_input(mlflow_faq_raw, context="faq-raw")
        mlflow.log_table(data=faq, artifact_file="faq_data.json")

        # Log raw dataset built from web4g
        mlflow_data_raw = mlflow.data.from_pandas(
            data.head(10),
            source=S3_PATH,
            name="web4g_data",
        )
        mlflow.log_input(mlflow_data_raw, context="pre-embedding")
        mlflow.log_table(data=data.head(10), artifact_file="web4g_data.json")

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        # Log environment necessary to reproduce the experiment
        current_dir = Path(".")
        FILES_TO_LOG = list(current_dir.glob("src/**/*.py")) + [PosixPath("run_build_database.py")]

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

        logger.info("Program ended with success.")


if __name__ == "__main__":
    run_build_database()
