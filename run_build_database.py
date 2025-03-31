import argparse
import os
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
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from src.config import set_config
from src.utils.prompt import similarity_search_instructions

from src.db_building.corpus import constructor_corpus
from src.db_building.document_chunker import parse_documents, chunk_documents
from src.utils.utils_vllm import get_model_from_env, get_model_max_len

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
parser.add_argument("--log_database_snapshot", action="store_true", help="Should we log database snapshot ? (default: False)")
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

url_database_client = config.get("QDRANT_URL")
api_key_database_client = config.get("QDRANT_API_KEY")
max_document_size = args.max_document_size
chunk_overlap = args.chunk_overlap
embedding_model = get_model_from_env("OPENAI_API_BASE_EMBEDDING", config)

parameters_database_construction = {
    "embedding_model": embedding_model,
    "QDRANT_URL_API": url_database_client,
    "chunking_strategy": args.chunking_strategy,
    "max_document_size": max_document_size,
    "chunk_overlap": chunk_overlap,
}
print(parameters_database_construction)
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

        model_max_len = get_model_max_len(embedding_model)
        unique_collection_name = f"{collection_name}_{run_id}"

        logger.info("Setting connection")
        client = QdrantClient(url=url_database_client, api_key=api_key_database_client, port="443", https="true")

        logger.info(f"Creating vector collection ({unique_collection_name})")
        client.create_collection(
            collection_name=unique_collection_name,
            vectors_config=VectorParams(size=model_max_len, distance=Distance.COSINE),
        )

        emb_model = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=config.get("OPENAI_API_BASE_EMBEDDING"),
            openai_api_key=config.get("OPENAI_API_KEY_EMBEDDING"),
        )

        # EMBEDDING DOCUMENTS IN VECTOR DATABASE -----------------------
        logger.info("Putting documents in vector database")

        db = QdrantVectorStore.from_documents(
            documents,
            emb_model,
            url=url_database_client,
            api_key=api_key_database_client,
            vector_name=embedding_model,
            prefer_grpc=False,
            port="443",
            https="true",
            collection_name=unique_collection_name,
            force_recreate=True,
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

        # LOGGING OTHER USEFUL THINGS --------------------------

        # Log raw dataset built from web4g
        mlflow_data_raw = mlflow.data.from_pandas(
            data.head(10),
            source=S3_PATH,
            name="web4g_data",
        )
        mlflow.log_input(mlflow_data_raw, context="pre-embedding")
        mlflow.log_table(data=data.head(10), artifact_file="web4g_data.json")

        # Log a result of a similarity search
        query = f"{similarity_search_instructions}\nQuery: Quels sont les chiffres du chômage en 2023 ?"
        mlflow.log_param("prompt_retriever", similarity_search_instructions)

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        retrieved_docs = retriever.invoke(query)
        result_list = []
        for doc in retrieved_docs:
            row = {"page_content": doc.page_content}
            row.update(doc.metadata)
            result_list.append(row)
        result = pd.DataFrame(result_list)
        mlflow.log_table(data=result, artifact_file="retrieved_documents.json")
        mlflow.log_param("question_asked", query)

        # Log parameters and metrics
        # mlflow.log_metric("number_documents", len(db_docs))

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
