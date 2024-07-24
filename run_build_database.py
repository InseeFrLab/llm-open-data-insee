import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PosixPath

import mlflow
import pandas as pd
import s3fs
from config import S3_BUCKET
from db_building import build_vector_database

# Global parameters
CHROMA_DB_LOCAL_DIRECTORY = "data/chroma_database"


def run_build_database(
    experiment_name: str,
    data_raw_s3_path: str,
    collection_name: str,
    emb_model_name: str,
    **kwargs,
):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": f"""https://{os.environ["AWS_S3_ENDPOINT"]}"""})

        db, df_raw = build_vector_database(
            data_path=data_raw_s3_path,
            persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
            embedding_model=emb_model_name,
            collection_name=collection_name,
            filesystem=fs,
            **kwargs,
        )

        # Log raw dataset built from web4g
        mlflow_data_raw = mlflow.data.from_pandas(df_raw.head(10), source=f"s3://{S3_BUCKET}/{data_raw_s3_path}", name="web4g_data")
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
        FILES_TO_LOG = (
            [PosixPath("src/build_database.py")] + list(current_dir.glob("src/db_building/*.py")) + list(current_dir.glob("src/config/*.py"))
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


if __name__ == "__main__":
    assert "MLFLOW_TRACKING_URI" in os.environ, "Please set the MLFLOW_TRACKING_URI environment variable."

    run_build_database(
        experiment_name="BUILD_CHROMA_TEST",
        data_raw_s3_path="data/raw_data/applishare_solr_joined.parquet",
        collection_name="insee_data",
        emb_model_name="OrdalieTech/Solon-embeddings-large-0.1",
        markdown_split=True,
        use_tokenizer_to_chunk=True,
        separators=["\n\n", "\n", ".", " ", ""],
        max_pages=150,
    )
