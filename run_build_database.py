import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path, PosixPath

import mlflow
import pandas as pd
import s3fs
import yaml

from src.db_building import build_vector_database, load_config

# Logging configuration
logger = logging.getLogger(__name__)


def run_build_database(config: Mapping[str, str]):
    mlflow.set_tracking_uri(config["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run():
        # Log parameters -------------------------------

        for arg_name, arg_value in locals().items():
            if arg_name == "kwargs":
                for key, value in arg_value.items():
                    mlflow.log_param(key, value)
            else:
                mlflow.log_param(arg_name, arg_value)

        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": f"https://{config['AWS_S3_ENDPOINT']}"})

        # Build database ------------------------------

        db, df_raw = build_vector_database(
            data_path=config["data_raw_s3_path"],
            persist_directory=config["chroma_db_local_dir"],
            collection_name=config["collection_name"],
            filesystem=fs,
            config=config,
        )

        logging.info("")

        # Log the parameters in a yaml file
        with open(f"{config['chroma_db_local_dir']}/parameters.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Move ChromaDB in a specific path in s3 -----------------------------

        hash_chroma = next(
            entry for entry in os.listdir(config["chroma_db_local_dir"]) if os.path.isdir(os.path.join(config["chroma_db_local_dir"], entry))
        )
        cmd = [
            "mc",
            "cp",
            "-r",
            f"{config['chroma_db_local_dir']}/",
            f"{config['chroma_db_s3_dir']}/{hash_chroma}/",
        ]
        subprocess.run(cmd, check=True)

        # Build database ------------------------------

        # Log raw dataset built from web4g
        mlflow_data_raw = mlflow.data.from_pandas(
            df_raw.head(10),
            source=config["raw_dataset_uri"],
            name="web4g_data",
        )
        mlflow.log_input(mlflow_data_raw, context="pre-embedding")
        mlflow.log_table(data=df_raw.head(10), artifact_file="web4g_data.json")

        # Log the vector database
        mlflow.log_artifacts(Path(config["chroma_db_local_dir"]), artifact_path="chroma")

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
            list(current_dir.glob("src/db_building/*.py")) + list(current_dir.glob("src/config/*.py")) + [PosixPath("run_build_database.py")]
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

        mlflow.log_param("chroma_path_s3_storage", config["path_chroma_stored_s3"])
        logger.info(f"Program ended with success, ChromaDB stored at location {config['path_chroma_stored_s3']}")


if __name__ == "__main__":
    config = load_config()
    assert config.has_option("DEFAULT", "MLFLOW_TRACKING_URI"), "Please set the MLFLOW_TRACKING_URI parameter (env variable or config file)."
    run_build_database(config["DEFAULT"])
