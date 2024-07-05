import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import s3fs
from config import COLLECTION_NAME, EMB_MODEL_NAME, S3_BUCKET
from db_building import build_vector_database

# Global parameters
EXPERIMENT_NAME = "BUILD_CHROMA_TEST"
MAX_NUMBER_PAGES = 20
CHROMA_DB_LOCAL_DIRECTORY = "data/chroma_database/chroma_test/"

# Check mlflow URL is defined
assert "MLFLOW_TRACKING_URI" in os.environ, "Please set the MLFLOW_TRACKING_URI environment variable."


# TODO: Cleaner le code
# TODO: Bien gérer le log des artifacts et tout ce qu'on veut dans MLflow
# TODO: Bien faire un script qui s'execute selon divers params
# TODO: Charger la db sur s3 (avec un paramètre ?)
# TODO: Revoir le chunking

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(EXPERIMENT_NAME)

# Build database
with mlflow.start_run() as run:
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": f"""https://{os.environ["AWS_S3_ENDPOINT"]}"""})

    db, data, chunk_infos = build_vector_database(
        data_path="data/raw_data/applishare_solr_joined.parquet",
        persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
        embedding_model=EMB_MODEL_NAME,
        collection_name=COLLECTION_NAME,
        filesystem=fs,
        max_pages=MAX_NUMBER_PAGES,
    )

    # logging.info(f"Test : {db.similarity_search("Quels sont les chiffres du chômages en 2023")}")

    # Log raw dataset built from web4g
    df_raw = pd.read_parquet(f"s3://{S3_BUCKET}/data/raw_data/applishare_solr_joined.parquet", filesystem=fs).head(10)
    mlflow_data_raw = mlflow.data.from_pandas(df_raw, source=f"s3://{S3_BUCKET}/data/raw_data/applishare_solr_joined.parquet", name="web4g_data")
    mlflow.log_input(mlflow_data_raw, context="pre-embedding")
    mlflow.log_table(data=df_raw, artifact_file="web4g_data.json")

    # Log the vector database
    mlflow.log_artifacts(Path(CHROMA_DB_LOCAL_DIRECTORY), artifact_path="chroma")

    # Log the first chunks of the vector database
    example_documents = pd.DataFrame(db.get()["documents"][:10], columns=["document"])
    mlflow.log_table(data=example_documents, artifact_file="example_documents.json")

    # Log a result of a similarity search
    logging.info(f"Test : {db.similarity_search("Quels sont les chiffres du chômages en 2023")}")

    # Log parameters and metrics
    mlflow.log_param("collection_name", COLLECTION_NAME)
    mlflow.log_param("number_pages", MAX_NUMBER_PAGES)
    mlflow.log_param("model_name", EMB_MODEL_NAME)
    mlflow.log_param("chunk_size", chunk_infos["chunk_size"])
    mlflow.log_param("chunk_overlap", chunk_infos["chunk_overlap"])
    mlflow.log_metric("number_documents", len(db.get()["documents"]))

    # Log all Python files in the current directory and its subdirectories
    current_dir = Path(".")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Copy Python files to the temporary directory, maintaining internal structure
        for file_path in current_dir.glob("**/*.py"):
            relative_path = file_path.relative_to(current_dir)
            destination_path = tmp_dir_path / relative_path.parent
            destination_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, destination_path)

        requirements_path = tmp_dir_path / "requirements.txt"

        # Generate requirements.txt using pipreqs
        subprocess.run(["pipreqs", str(tmp_dir_path)], check=True)

        # Log all Python files to MLflow artifact
        mlflow.log_artifacts(tmp_dir, artifact_path="environment")
