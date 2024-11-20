import logging
import os
import tempfile
from collections.abc import Mapping
from typing import Any, Protocol

import chromadb
import mlflow
import pandas as pd
import s3fs
import yaml
from langchain.schema import Document
from langchain_chroma import Chroma

from src.config import RAGConfig
from src.utils import compare_params

from .build_database import build_vector_database, load_vector_database_from_local

logger = logging.getLogger(__name__)


class LocalLoadingConfig(Protocol):
    chroma_db_local_path: str
    collection_name: str


class MLFlowLoadingConfig(Protocol):
    mlflow_load_artifacts: bool
    mlflow_run_id: str | None


class LoadingConfig(LocalLoadingConfig, MLFlowLoadingConfig, Protocol):
    pass


# LOADING VECTOR DATABASE FROM DIRECTORY -------------------------


def vector_database_available_from_local(
    persist_directory: str | None = None, config: LocalLoadingConfig = RAGConfig()
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
    mlflow_run_id: str | None = None, config: MLFlowLoadingConfig = RAGConfig()
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


def load_vector_database_from_mlflow(
    run_id: str | None = None, config: Mapping[str, Any] = vars(RAGConfig())
) -> Chroma:
    """ """
    run_id = vector_database_available_from_mlflow(run_id, config=vars(config))
    if run_id is None:
        raise FileNotFoundError("No database found in S3 storage")
    return _load_vector_database_from_mlflow(run_id, config)


def _load_vector_database_from_mlflow(run_id: str, config: Mapping[str, Any] = vars(RAGConfig())) -> Chroma:
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


def vector_database_available_from_s3(
    filesystem: s3fs.S3FileSystem, config: Mapping[str, Any] = vars(RAGConfig())
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

    missing_keys = [key for key in required_keys if not config.get(key)]
    if missing_keys:
        logger.warn(f"Missing possibly required arguments: {', '.join(missing_keys)}")

    logger.info("Searching for database with the following parameters:")
    for key in required_keys:
        logger.info(f"  {key}: {config.get(key)}")

    db_path_prefix = f"{config['s3_bucket']}/data/chroma_database/{config['emb_model']}"

    if not filesystem.exists(db_path_prefix):
        logger.info(f"Expected S3 folder for database does not exist: {db_path_prefix}")
        return None

    logger.info(f"Looking for database in S3 folder: {db_path_prefix}")
    for db_path in filesystem.ls(db_path_prefix):
        with filesystem.open(f"{db_path}/parameters.yaml") as f:
            params = yaml.safe_load(f)
            different_params = compare_params(config, params)
        if not different_params:
            logger.info("Found database with matching parameters!")
            return db_path
        logger.info(f"Non matching parameters {different_params}: {params}")
    return None


def load_vector_database_from_s3(
    filesystem: s3fs.S3FileSystem, config: Mapping[str, Any] = vars(RAGConfig())
) -> Chroma:
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


def _load_vector_database_from_s3(
    db_path: str, filesystem: s3fs.S3FileSystem, config: Mapping[str, Any] = vars(RAGConfig())
) -> Chroma:
    with tempfile.TemporaryDirectory() as temp_dir:
        filesystem.get(db_path, temp_dir, recursive=True)
        return load_vector_database_from_local(temp_dir, config)


# LOADING FROM ANY SOURCE AVAILABLE ----------------------------------------


def load_vector_database(
    filesystem: s3fs.S3FileSystem,
    config: LoadingConfig = RAGConfig(),
    allow_local: bool = True,
    allow_mlflow: bool = True,
    allow_s3: bool = True,
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
        return load_vector_database_from_local(config=vars(config))

    if allow_mlflow:
        mlflow_run_id = vector_database_available_from_mlflow(config=config)
        if mlflow_run_id:
            logger.info("Loading database from MLFlow artifacts")
            local_path = _download_mlflow_artifacts_if_exists(run_id=mlflow_run_id)
            return load_vector_database_from_local(local_path, vars(config))

    if allow_s3:
        s3path = vector_database_available_from_s3(filesystem=filesystem, config=vars(config))
        if s3path is not None:
            logger.info("Loading database from S3")
            return _load_vector_database_from_s3(s3path, filesystem, vars(config))

    logger.info("No database found")
    return None


def build_or_load_vector_database(
    filesystem: s3fs.S3FileSystem,
    config: LoadingConfig = RAGConfig(),
    allow_local: bool = True,
    allow_mlflow: bool = True,
    allow_s3: bool = True,
    document_database: tuple[pd.DataFrame, list[Document]] | None = None,
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
        filesystem, config=vars(config), return_none_on_fail=True, document_database=document_database
    )
