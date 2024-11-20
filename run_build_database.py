import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PosixPath

import mlflow
import pandas as pd
import s3fs
import yaml

from src.config import RAGConfig, process_args, simple_argparser
from src.db_building import build_or_load_document_database, build_vector_database, load_vector_database

# Logging configuration
logger = logging.getLogger(__name__)


def run_build_database() -> None:
    config = RAGConfig()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run():
        # Logging the full configuration to mlflow
        mlflow.log_params(dict(config))

        filesystem = s3fs.S3FileSystem(endpoint_url=config.s3_endpoint_url)

        # Log the parameters in a yaml file
        os.makedirs(config.chroma_db_local_path, exist_ok=True)
        with open(f"{config.chroma_db_local_path}/parameters.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Load or build the document database
        df, all_splits = build_or_load_document_database(filesystem, vars(config))

        # Load or build the vector database
        db = load_vector_database(filesystem, config)
        if db is None:
            db = build_vector_database(
                filesystem, config=vars(config), return_none_on_fail=True, document_database=(df, all_splits)
            )
            # Move ChromaDB in a specific path in s3
            hash_chroma = next(
                entry
                for entry in os.listdir(config.chroma_db_local_path)
                if os.path.isdir(os.path.join(config.chroma_db_local_path, entry))
            )
            logger.info(f"Uploading Chroma database ({hash_chroma}) to s3: {config.chroma_db_local_path}")
            cmd = (
                "mc",
                "cp",
                "-r",
                f"{config.chroma_db_local_path}/",
                f"{config.chroma_db_s3_dir}/{hash_chroma}/",
            )
            subprocess.run(cmd, check=True)

            # Log the newly generated vector database unless it was already loaded from an other run ID
            logger.info(f"Logging to MLFlow ({hash_chroma}) to s3: {config.chroma_db_local_path}")
            if not (config.mlflow_run_id and config.mlflow_load_artifacts):
                mlflow.log_artifacts(config.chroma_db_local_path, artifact_path="chroma")

        # Log raw dataset built from web4g
        mlflow_data_raw = mlflow.data.from_pandas(
            df.head(10),
            source=config.raw_dataset_uri,
            name="web4g_data",
        )
        mlflow.log_input(mlflow_data_raw, context="pre-embedding")
        mlflow.log_table(data=df.head(10), artifact_file="web4g_data.json")

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
        FILES_TO_LOG = (
            list(current_dir.glob("src/db_building/*.py"))
            + list(current_dir.glob("src/config/*.py"))
            + [PosixPath("run_build_database.py")]
        )

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
        logger.info(f"ChromaDB stored at location {config['path_chroma_stored_s3']}")


if __name__ == "__main__":
    process_args(simple_argparser())
    assert RAGConfig().mlflow_tracking_uri is not None, "Please set the MLFLOW_TRACKING_URI parameter"
    run_build_database()
