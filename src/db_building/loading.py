import logging
import os
import tempfile

import chromadb
import mlflow
import pandas as pd
import s3fs
import yaml
from langchain.schema import Document
from langchain_chroma import Chroma
from pydantic import BaseModel

from src.config import Configurable, DefaultFullConfig, FullConfig
from src.utils import compare_params

from .build_database import build_vector_database, load_vector_database_from_local

logger = logging.getLogger(__name__)


class LocalLoadingConfig(BaseModel):
    chroma_db_local_path: str
    collection_name: str


class MLFlowLoadingConfig(BaseModel):
    mlflow_load_artifacts: bool
    mlflow_run_id: str | None


class LoadingConfig(LocalLoadingConfig, MLFlowLoadingConfig, BaseModel):
    pass


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


@Configurable()
def load_vector_database_from_mlflow(
    run_id: str | None = None, config: MLFlowLoadingConfig = DefaultFullConfig()
) -> Chroma:
    """ """
    run_id = vector_database_available_from_mlflow(run_id, config=config)
    if run_id is None:
        raise FileNotFoundError("No database found in S3 storage")
    return _load_vector_database_from_mlflow(run_id, config)


@Configurable()
def _load_vector_database_from_mlflow(run_id: str, config: MLFlowLoadingConfig = DefaultFullConfig()) -> Chroma:
    logger.info("Loading database from MLFlow artifacts")
    local_path = _download_mlflow_artifacts_if_exists(run_id=run_id)
    return load_vector_database_from_local(local_path, config)


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
        "use_tokenizer_to_chunk",
        "separators",
        "emb_model",
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

    s3_folder = f"{config.chroma_db_s3_path}/{config.emb_model}"

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


@Configurable()
def load_vector_database_from_s3(filesystem: s3fs.S3FileSystem, config: FullConfig = DefaultFullConfig()) -> Chroma:
    """
    Load vector database from S3 storage.

    Args:
    filesystem: S3 file system
    config: configuration object with the database parameters (embedding model, etc.)

    Returns:
    The loaded Chroma database
    """
    s3path = vector_database_available_from_s3(filesystem, config)
    if s3path is None:
        raise FileNotFoundError("No database found in S3 storage")
    return _load_vector_database_from_s3(s3path, filesystem, config)


@Configurable()
def _load_vector_database_from_s3(
    db_path: str,
    filesystem: s3fs.S3FileSystem,
    config: LoadingConfig = DefaultFullConfig(),
) -> Chroma:
    with tempfile.TemporaryDirectory() as temp_dir:
        filesystem.get(db_path, temp_dir, recursive=True)
        return load_vector_database_from_local(temp_dir, config)


# LOADING FROM ANY SOURCE AVAILABLE ----------------------------------------


@Configurable()
def load_vector_database(
    filesystem: s3fs.S3FileSystem,
    allow_local: bool = True,
    allow_mlflow: bool = True,
    allow_s3: bool = True,
    config: LoadingConfig = DefaultFullConfig(),
) -> Chroma | None:
    """
    Load a vector database from either a local session, MLFlow artifacts or S3 storage.

    Args:
    filesystem: The filesystem object for interacting with S3.
    config: Extra configuration for the database loading
    allow_local: Allow to load database from an existing local session
    allow_mlflow: Allow to load database from a previous MLFlow run
    allow_s3: Allow to load database from an S3 parquet file

    Returns:
    The loaded database object or None if an error occurred or no database could be found
    """
    if allow_local and vector_database_available_from_local(config=config):
        logging.info("Loading database from local session")
        return load_vector_database_from_local(config=config)

    if allow_mlflow:
        mlflow_run_id = vector_database_available_from_mlflow(config=config)
        if mlflow_run_id:
            logger.info("Loading database from MLFlow artifacts")
            local_path = _download_mlflow_artifacts_if_exists(run_id=mlflow_run_id)
            return load_vector_database_from_local(local_path, config)

    if allow_s3:
        s3path = vector_database_available_from_s3(filesystem=filesystem, config=config)
        if s3path is not None:
            logger.info("Loading database from S3")
            return _load_vector_database_from_s3(s3path, filesystem, config)

    logger.info("No database found")
    return None


@Configurable()
def build_or_load_vector_database(
    filesystem: s3fs.S3FileSystem,
    allow_local: bool = True,
    allow_mlflow: bool = True,
    allow_s3: bool = True,
    document_database: tuple[pd.DataFrame, list[Document]] | None = None,
    config: LoadingConfig = DefaultFullConfig(),
) -> Chroma | None:
    """
    Load a vector database from either a local session, MLFlow artifacts or S3 storage.
    If no database is available, builds it from documents.

    Args:
    filesystem: The filesystem object for interacting with S3.
    config: Extra configuration for the database loading
    allow_local: Allow to load database from an existing local session
    allow_mlflow: Allow to load database from a previous MLFlow run
    allow_s3: Allow to load database from an S3 parquet file
    allow_build: Allow to rebuild the database from the documents

    Returns:
    The loaded Chroma database object or None if no database could be found and an error occurred during build
    """
    return load_vector_database(filesystem, config, allow_local, allow_mlflow, allow_s3) or build_vector_database(
        filesystem, config, return_none_on_fail=True, document_database=document_database
    )
