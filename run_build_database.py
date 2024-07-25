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
import yaml

from src.config import (
    CHROMA_DB_LOCAL_DIRECTORY,
    S3_BUCKET,
)
from src.db_building import build_vector_database

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

parser = argparse.ArgumentParser(description="Chroma building parameters")
parser.add_argument(
    "--experiment_name",
    type=str,
    default="default",
    help="""
    Name of the experiment.
    """,
)
parser.add_argument(
    "--data_raw_s3_path",
    type=str,
    default="data/raw_data/applishare_solr_joined.parquet",
    help="""
    Path to the raw data.
    Default to data/raw_data/applishare_solr_joined.parquet
    """,
)
parser.add_argument(
    "--collection_name",
    type=str,
    default="insee_data",
    help="""
    Collection name.
    Default to insee_data
    """,
)
parser.add_argument(
    "--markdown_split",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="""
    Should we use a markdown split ?
    --markdown_split yields True and --no-markdown_split yields False
    """,
)
parser.add_argument(
    "--use_tokenizer_to_chunk",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="""
    Should we use the tokenizer of the embedding model to chunk ?
    --use_tokenizer_to_chunk yields True and --no-use_tokenizer_to_chunk yields False
    """,
)
parser.add_argument(
    "--separators",
    type=str_to_list,
    default=r"['\n\n', '\n', '.', ' ', '']",
    help="List separators to split the text",
)
parser.add_argument(
    "--embedding_model",
    type=str,
    default="OrdalieTech/Solon-embeddings-large-0.1",
    help="""
    Embedding model.
    Should be a huggingface model.
    Defaults to OrdalieTech/Solon-embeddings-large-0.1
    """,
)
parser.add_argument(
    "--max_pages",
    type=int,
    default=None,
    help="""
    Maximum number of pages to use for the vector database.
    """,
)
parser.add_argument(
    "--chunk_size",
    type=str,
    default=None,
    help="""
    Chunk size
    """,
)
parser.add_argument(
    "--chunk_overlap",
    type=str,
    default=None,
    help="""
    Chunk overlap
    """,
)

args = parser.parse_args()


def run_build_database(
    experiment_name: str,
    data_raw_s3_path: str,
    collection_name: str,
    **kwargs,
):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)

    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": f"""https://{os.environ["AWS_S3_ENDPOINT"]}"""})

    with mlflow.start_run():
        # Log parameters
        for key, value in kwargs.items():
            mlflow.log_param(key, value)

        db, df_raw = build_vector_database(
            data_path=data_raw_s3_path,
            persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
            collection_name=collection_name,
            filesystem=fs,
            **kwargs,
        )

        # Log raw dataset built from web4g
        mlflow_data_raw = mlflow.data.from_pandas(
            df_raw.head(10),
            source=f"s3://{S3_BUCKET}/{data_raw_s3_path}",
            name="web4g_data",
        )
        mlflow.log_input(mlflow_data_raw, context="pre-embedding")
        mlflow.log_table(data=df_raw.head(10), artifact_file="web4g_data.json")

        # Log the vector database
        mlflow.log_artifacts(Path(CHROMA_DB_LOCAL_DIRECTORY), artifact_path="chroma")

        # Log the first chunks of the vector database
        db_docs = db.get()["documents"]
        example_documents = pd.DataFrame(db_docs[:10], columns=["document"])
        mlflow.log_table(data=example_documents, artifact_file="example_documents.json")

        # Log a result of a similarity search
        query = "Quels sont les chiffres du chômages en 2023 ?"
        retrieved_docs = db.similarity_search(query, k=5)

        result_list = []
        for doc in retrieved_docs:
            row = {"page_content": doc.page_content}
            row.update(doc.metadata)
            result_list.append(row)
        result = pd.DataFrame(result_list)
        mlflow.log_table(data=result, artifact_file="retrieved_documents.json")
        mlflow.log_param("question_asked", query)

        # Log parameters and metrics
        mlflow.log_metric("number_documents", len(db_docs))

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

        # Log the parameters in a yaml file
        with open(f"{CHROMA_DB_LOCAL_DIRECTORY}/parameters.yaml", "w") as f:
            params = {
                "data_raw_s3_path": data_raw_s3_path,
                "collection_name": collection_name,
            } | kwargs
            yaml.dump(params, f, default_flow_style=False)

        # Move ChromaBD in a specific path in s3
        cmd = [
            "mc",
            "cp",
            "-r",
            CHROMA_DB_LOCAL_DIRECTORY,
            f"s3/{S3_BUCKET}/chroma_database/{kwargs.get("embedding_model")}/",
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    assert "MLFLOW_TRACKING_URI" in os.environ, "Please set the MLFLOW_TRACKING_URI environment variable."

    args = parser.parse_args()

    run_build_database(
        **vars(args),
    )
