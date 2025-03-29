import logging
import os
import tempfile

import chromadb
import mlflow
import s3fs
import yaml
from pydantic import BaseModel

from src.config import Configurable, DefaultFullConfig, FullConfig
from src.utils import compare_params

logger = logging.getLogger(__name__)


# LOADING VECTOR DATABASE FROM DIRECTORY -------------------------


def vector_database_available_from_local(
    persist_directory: str | None = None, config: LocalLoadingConfig = DefaultFullConfig()
) -> bool:
    """
    Args:
        paths: Paths to look into. config.chroma_db_local_path
    Returns:
        A path
    """
    if persist_directory is None:
        persist_directory = config.chroma_db_local_path
    if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        return False
    client = chromadb.PersistentClient(path=persist_directory)
    return config.collection_name in [c.name for c in client.list_collections()]


# LOADING FROM MLFLOW ARTIFACTS ----------------------------------------


def vector_database_available_from_mlflow(
    mlflow_run_id: str | None = None, config: MLFlowLoadingConfig = DefaultFullConfig()
) -> str | None:
    """
    Checks that the provided MLFlow run ID has a Chroma vector database in its artifacts.

    Args:
        mlflow_run_id: An MLFlow run ID to check
        config: configuration object with the database parameters (embedding model, etc.)

    Returns:
        An MLFlow run ID whose artifacts contain the found vector database matching the provided parameters
        or None
    """
    if mlflow_run_id is None and config.mlflow_load_artifacts and config.mlflow_run_id:
        mlflow_run_id = config.mlflow_run_id
    if mlflow_run_id and mlflow.artifacts.list_artifacts(run_id=mlflow_run_id, artifact_path="chroma"):
        return mlflow_run_id
    else:
        return None


def _download_mlflow_artifacts_if_exists(run_id: str, dst_path: str | None = None, force: bool = False) -> str:
    """
    Args:
        run_id: MLFlow run id
        dst_path: Local path where artifacts will be downloaded to. Defaults to a temporary folder.
        force: Overwrite already existing files ?

    Returns:
        The path where artifacts were loaded.
    """
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


@Configurable()
def vector_database_available_from_s3(
    filesystem: s3fs.S3FileSystem, config: FullConfig = DefaultFullConfig()
) -> str | None:
    """
    Finds a path in S3 storage containing a Chroma vector database.

    Args:
        filesystem: S3 file system
        config: configuration object with the database parameters (embedding model, etc.)

    Returns:
        Path to the S3 folder containing the found vector database matching the provided parameters
        or None if no such database could be found
    """
    required_keys = [
        "data_raw_s3_path",
        "collection_name",
        "markdown_split",
        "separators",
        "embedding_model",
        "chunk_size",
        "chunk_overlap",
    ]
    config_dict = vars(config)
    missing_keys = [key for key in required_keys if not config_dict.get(key)]
    if missing_keys:
        logger.warn(f"Missing possibly required arguments: {', '.join(missing_keys)}")

    logger.info("Searching for database with the following parameters:")
    for key in required_keys:
        logger.info(f"  {key}: {config_dict.get(key)}")

    s3_folder = f"{config.chroma_db_s3_path}/"

    if not filesystem.exists(s3_folder):
        logger.info(f"Expected S3 folder for database does not exist: {s3_folder}")
        return None

    logger.info(f"Looking for database in S3 folder: {s3_folder}")
    for s3_db_path in filesystem.ls(s3_folder):
        with filesystem.open(f"{s3_db_path}/parameters.yaml") as f:
            params = yaml.safe_load(f)
            different_params = compare_params(config, params)
        if not different_params:
            logger.info("Found database with matching parameters!")
            return s3_db_path
        logger.info(f"Non matching parameters {different_params}: {params}")
    return None
