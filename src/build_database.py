from pathlib import Path
import os
import subprocess
import tempfile
import shutil

import mlflow
import pandas as pd
from config import COLLECTION_NAME, EMB_MODEL_NAME
from db_building import build_database_from_csv
from doc_building import compute_autokonenizer_chunk_size

# Global parameters
EXPERIMENT_NAME = "BUILD_CHROMA_TEST"
MAX_NUMBER_PAGES = 100
CHROMA_DB_LOCAL_DIRECTORY = "data/chroma_database/chroma_test/"

# Rustine temporaire
os.environ["MLFLOW_TRACKING_URI"] = "https://projet-llm-insee-open-data-mlflow.user.lab.sspcloud.fr"

# Check mlflow URL is defined
assert (
    "MLFLOW_TRACKING_URI" in os.environ
), "Please set the MLFLOW_TRACKING_URI environment variable."

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(EXPERIMENT_NAME)

# Build database
with mlflow.start_run() as run:
    # extract chunk size and overlap
    _, chunk_size, chunk_overlap = compute_autokonenizer_chunk_size(EMB_MODEL_NAME)

    db = build_database_from_csv(
        "data_complete.csv",
        persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
        max_pages=MAX_NUMBER_PAGES,
    )
    db.similarity_search("Quels sont les chiffres du chômages en 2023")

    # Create a dataset instance to record a few things in MLFlow
    dataset = mlflow.data.from_pandas(
        pd.read_csv("data_complete.csv").head(10), source="data_complete.csv"
    )
    first_documents = pd.DataFrame(db.get()["documents"][:10], columns=["document"])

    mlflow.log_input(dataset, context="sample_web4g")
    mlflow.log_param("collection_name", COLLECTION_NAME)
    mlflow.log_param("number_pages", MAX_NUMBER_PAGES)
    mlflow.log_param("model_name", EMB_MODEL_NAME)
    mlflow.log_param("chunk_size", chunk_size)
    mlflow.log_param("chunk_overlap", chunk_overlap)
    mlflow.log_metric("number_documents", len(db.get()["documents"]))
    mlflow.log_table(data=first_documents, artifact_file="example_documents.json")

    # Log all Python files in the current directory and its subdirectories
    current_dir = Path(".")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Copy Python files to temporary directory maintaining internal structure
        for file_path in current_dir.glob("**/*.py"):
            relative_path = file_path.relative_to(current_dir)
            destination_path = tmp_dir / relative_path.parent
            destination_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, destination_path)

        requirements_path = f"{tmp_dir}/requirements.txt"
        subprocess.run(["pip", "freeze"], stdout=open(requirements_path, "w"))

        # Log all Python files to MLflow artifact
        mlflow.log_artifacts(tmp_dir, artifact_path="environment")

    chroma_dir = Path(CHROMA_DB_LOCAL_DIRECTORY)
    mlflow.log_artifacts(chroma_dir, artifact_path="chroma")
