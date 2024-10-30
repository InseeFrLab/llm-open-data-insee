import logging
import os
import tempfile
import warnings
from collections.abc import Mapping
from typing import Any

import mlflow
import s3fs
import yaml
from langchain_community.vectorstores import Chroma

from src.config import default_config

from .build_database import reload_database_from_local_dir

logger = logging.getLogger(__name__)


def load_vector_database(filesystem: s3fs.S3FileSystem, config: Mapping[str, Any] = default_config) -> Chroma | None:
    """
    Loads a vector database from S3 or a local path based on the provided parameters.

    Parameters:
    filesystem : s3fs.S3FileSystem
        The filesystem object for interacting with S3.
    config : Mapping[str, Any]
        Keyword arguments for configuring the database loading.
        - database_run_id (str, optional): If provided, download artifacts using MLflow.
        - Other parameters as required by the specific use case.

    Returns:
    Optional[Chroma]
        The loaded database object or None if an error occurred.
    """
    try:
        if (
            config["mlflow_run_id"]
            and config["mlflow_load_artifacts"]
            and mlflow.artifacts.list_artifacts(run_id=config["mlflow_run_id"], artifact_path="chroma")
        ):
            local_path = _download_mlflow_artifacts_if_exists(run_id=config["mlflow_run_id"])
            return reload_database_from_local_dir(local_path, config)
        else:
            return _load_database_from_s3(filesystem, config)
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
    return None


# DOWNLOADING ARTIFACTS FROM MLFLOW ----------------------------------------


def _download_mlflow_artifacts_if_exists(run_id: str, dst_path: str | None = None, force: bool = False) -> str:
    # Construct the destination path
    if dst_path is None:
        tmpdir = tempfile.gettempdir()
        dst_path = os.path.join(tmpdir, "mlflow", run_id, "chroma")

    # Check if dst_path exists and force is False
    if os.path.exists(dst_path) and not force:
        logger.info(f"Destination path {dst_path} exists and `force` is False. Skipping download.")
        local_path = f"{dst_path}/chroma"
    else:
        # If dst_path doesn't exist or force is True,
        # ensure the directory exists and download artifacts
        os.makedirs(dst_path, exist_ok=True)

        # Download artifacts to the specific dst_path
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="chroma", dst_path=dst_path)

        logger.info(f"Artifacts downloaded to: {local_path}")

    return local_path


# LOADING FROM S3 ----------------------------------------


def _load_database_from_s3(filesystem: s3fs.S3FileSystem, config: Mapping[str, Any] = default_config) -> Chroma:
    """Helper function to load database from S3 based on provided parameters."""
    required_keys = [
        "data_raw_s3_path",
        "collection_name",
        "markdown_split",
        "use_tokenizer_to_chunk",
        "separators",
        "emb_model",
        "chunk_size",
        "chunk_overlap",
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        warnings.warn(
            f"Missing possibly required arguments: {', '.join(missing_keys)}",
            stacklevel=2,
        )

    logger.info("Searching for database with the following parameters:")
    for key in required_keys:
        if key in config:
            logger.info(key, "-", config[key])

    db_path_prefix = f"{config['s3_bucket']}/data/chroma_database/{config['emb_model']}"

    logger.info(f"Checking if a database has been stored at location '{db_path_prefix}'")

    if not filesystem.exists(db_path_prefix):
        raise FileNotFoundError(f"Database with model '{config['emb_model']}' not found")

    for db_path in filesystem.ls(db_path_prefix):
        with filesystem.open(f"{db_path}/parameters.yaml") as f:
            params = yaml.safe_load(f)
            different_params = compare_params(config, params)
            logger.info(params)

        if not different_params:
            return _reload_database_from_s3(filesystem, db_path, config)

    raise FileNotFoundError(f"Database with parameters {config} not found")


def _reload_database_from_s3(filesystem: s3fs.S3FileSystem, db_path: str, config: Mapping[str, Any] = default_config) -> Chroma:
    """Helper function to reload database from S3 to a local temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filesystem.get(db_path, temp_dir, recursive=True)
        return reload_database_from_local_dir(temp_dir, config)


def compare_params(big_dict, small_dict):
    # Initialize a list to hold keys with different values
    different_keys = []

    # Iterate over the keys in the smaller dictionary
    for key in small_dict:
        # Check if the key exists in the bigger dictionary and if the values are different
        if key in big_dict and big_dict[key] != small_dict[key]:
            different_keys.append(key)

    return different_keys
