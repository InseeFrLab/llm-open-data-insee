import logging
import tempfile
import warnings

import mlflow
import s3fs
import yaml
from langchain_community.vectorstores import Chroma

from src.config import S3_BUCKET

from .build_database import reload_database_from_local_dir


def load_vector_database(filesystem: s3fs.S3FileSystem, **kwargs) -> Chroma:
    """
    Loads a vector database from S3 or a local path based on the provided parameters.

    Parameters:
    - filesystem (s3fs.S3FileSystem): The filesystem object for interacting with S3.
    - kwargs (dict): Keyword arguments for configuring the database loading.
        - database_run_id (str, optional): If provided, download artifacts using MLflow.
        - Other parameters as required by the specific use case.

    Returns:
    - db (object, optional): The loaded database object or None if an error occurred.
    """
    try:
        if kwargs.get("database_run_id") is not None:
            return _load_database_from_mlflow(kwargs["database_run_id"])
        else:
            return _load_database_from_s3(filesystem, kwargs)
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
    return None


def _load_database_from_mlflow(run_id: str) -> Chroma:
    """Helper function to load database from MLflow artifacts."""
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="chroma")
    run_params = mlflow.get_run(run_id).data.params
    db = reload_database_from_local_dir(
        embed_model_name=run_params["embedding_model"],
        collection_name=run_params["collection_name"],
        persist_directory=local_path,
        embed_device=run_params["embedding_device"],
    )
    return db


def _load_database_from_s3(filesystem: s3fs.S3FileSystem, kwargs: dict[str, str]) -> Chroma:
    """Helper function to load database from S3 based on provided parameters."""
    required_keys = [
        "data_raw_s3_path",
        "collection_name",
        "markdown_split",
        "use_tokenizer_to_chunk",
        "separators",
        "embedding_model",
        "chunk_size",
        "chunk_overlap",
    ]

    missing_keys = [key for key in required_keys if key not in kwargs]

    if missing_keys:
        warnings.warn(f"Missing possibly required arguments: {', '.join(missing_keys)}", stacklevel=2)

    logging.info(f"Searching for database with the following parameters: {kwargs}")

    db_path_prefix = f"{S3_BUCKET}/data/chroma_database/{kwargs.get('embedding_model')}"

    if not filesystem.exists(db_path_prefix):
        raise FileNotFoundError(f"Database with model '{kwargs.get('embedding_model')}' not found")

    for db_path in filesystem.ls(db_path_prefix):
        with filesystem.open(f"{db_path}/parameters.yaml") as f:
            params = yaml.safe_load(f)
            different_params = compare_params(kwargs, params)

        if not different_params:
            return _reload_database_from_s3(filesystem, db_path, kwargs)

    raise FileNotFoundError(f"Database with parameters {kwargs} not found")


def _reload_database_from_s3(filesystem: s3fs.S3FileSystem, db_path: str, kwargs: dict[str, str]) -> Chroma:
    """Helper function to reload database from S3 to a local temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filesystem.get(f"{db_path}", temp_dir, recursive=True)

        db = reload_database_from_local_dir(
            embed_model_name=kwargs["embedding_model"],
            collection_name=kwargs["collection_name"],
            persist_directory=temp_dir,
            embed_device=kwargs["embedding_device"],
        )
        return db


def compare_params(big_dict, small_dict):
    # Initialize a list to hold keys with different values
    different_keys = []

    # Iterate over the keys in the smaller dictionary
    for key in small_dict:
        # Check if the key exists in the bigger dictionary and if the values are different
        if key in big_dict and big_dict[key] != small_dict[key]:
            different_keys.append(key)

    return different_keys
