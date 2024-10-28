import logging
import typing as t
from typing import Mapping, Any

import jsonlines
import pandas as pd
import s3fs
from langchain.schema import Document

from .document_chunker import chunk_documents
from .utils_db import parse_xmls

logger = logging.getLogger(__name__)

# CHUNKING ALL DATASET ---------------------------------------------------


def preprocess_and_store_data(
    config: Mapping[str, Any],
    filesystem: s3fs.S3FileSystem
) -> tuple[pd.DataFrame, t.Iterable[Document]]:
    """
    Process data from S3, chunk the documents, and store them in an intermediate location.

    Parameters:
    - config: Configuration for data processing and chunking (e.g., 'max_pages', 'chunk_overlap', 'chunk_size').
    - filesystem: s3fs.S3FileSystem object for interacting with S3.
    - location_dataset: dict, paths to the main data files in the S3 bucket.

    Returns:
    - Tuple containing the processed DataFrame and chunked documents (list of Document objects).
    """

    # Handle data loading, parsing, and splitting
    df, all_splits = _preprocess_data(config, filesystem)

    logger.info("Saving chunked documents in an intermediate location")

    # Create the storage path for intermediate data
    data_intermediate_storage = _s3_path_intermediate_collection(config)

    # Save chunked documents to JSONL using S3 and s3fs
    save_docs_to_jsonl(all_splits, data_intermediate_storage, filesystem)

    # Save the DataFrame as a Parquet file in the same directory
    parquet_file_path = data_intermediate_storage.replace("docs.jsonl", "corpus.parquet")
    df.loc[:, ~df.columns.isin(["dateDiffusion"])] = df.loc[:, ~df.columns.isin(["dateDiffusion"])].astype(str)
    df.to_parquet(parquet_file_path, filesystem=filesystem, index=False)
    logger.info(f"DataFrame saved to {parquet_file_path}")

    return df, all_splits


def _preprocess_data(
    config: Mapping[str, Any],
    filesystem: s3fs.S3FileSystem
) -> tuple[pd.DataFrame, list[Document]]:
    """
    Process and merge data from multiple parquet sources, parse XML content,
    and split the documents into chunks for embedding.

    Parameters:
    - s3_bucket: str, the S3 bucket name.
    - filesystem: object, the filesystem handler (e.g., s3fs).
    - location_dataset: dict, path to the main data files in the S3 bucket.
    - kwargs: optional keyword arguments to control parser behavior, including 'max_pages' for limiting the number of rows.

    Returns:
    - all_splits: list or DataFrame containing the processed and chunked documents.
    """

    logger.info(f"Input data extracted from s3://{config['s3_bucket']}")

    # Load main data from parquet file
    data = pd.read_parquet(config['rawdata_web4g_uri'], filesystem=filesystem)

    # Limit the number of pages if specified
    if config.get("max_pages") is not None:
        data = data.head(config.get("max_pages"))

    # Parse the XML content
    parsed_pages = parse_xmls(data)

    # Merge parsed XML data with the original data
    df = data.set_index("id").merge(pd.DataFrame(parsed_pages), left_index=True, right_index=True)

    # Select relevant columns
    df = df[
        [
            "titre",
            "categorie",
            "url",
            "dateDiffusion",
            "theme",
            "collection",
            "libelleAffichageGeo",
            "content",
        ]
    ]

    # Load additional RMES data
    data_rmes = pd.read_parquet(config['rawdata_rmes_uri'], filesystem=filesystem)

    # Concatenate the original data with the RMES data
    df = pd.concat([df, data_rmes])

    # Fill NaN values with empty strings (for compatibility with Chroma metadata)
    df = df.fillna(value="")

    # Chunk the documents (using tokenizer if specified in kwargs)
    all_splits = chunk_documents(
        data=df,
        embedding_model=config['emb_model'],
        markdown_split=config['markdown_split'],
        use_tokenizer_to_chunk=config['use_tokenizer_to_chunk'],
        chunk_size=config.get('chunk_size'),
        chunk_overlap=config.get('chunk_overlap'),
        separators=config.get('separators')
    )

    return df, all_splits


# RECHUNKING OR LOADING FROM DATA STORE --------------------------


def build_or_use_from_cache(
    config: Mapping[str, Any],
    filesystem: s3fs.S3FileSystem
) -> tuple[pd.DataFrame, t.Iterable[Document]]:
    """
    Either load the chunked documents and DataFrame from cache or process and store the data.

    Parameters:
    - filesystem: s3fs.S3FileSystem object for interacting with S3.
    - s3_bucket: str, the name of the S3 bucket.
    - location_dataset: dict, paths to the main data files in the S3 bucket.
    - force_rebuild: bool, if True, will force data processing and storage even if cache exists.
    - kwargs: Optional keyword arguments for data processing and chunking (e.g., 'max_pages', 'chunk_overlap', 'chunk_size').

    Returns:
    - Tuple containing the processed DataFrame and chunked documents (list of Document objects).
    """

    # Create the storage path for intermediate data
    data_intermediate_storage = _s3_path_intermediate_collection(config)

    # Create the Parquet file path (corpus.parquet)
    parquet_file_path = data_intermediate_storage.replace("docs.jsonl", "corpus.parquet")

    # Check if we should use the cached data
    if not config.get('force_rebuild'):
        try:
            # Attempt to load the cached documents from S3
            logger.info(f"Attempting to load chunked documents from {data_intermediate_storage}")
            all_splits = load_docs_from_jsonl(data_intermediate_storage, filesystem)
            logger.info("Loaded chunked documents from cache")

            # Attempt to load the cached DataFrame from S3
            logger.info(f"Attempting to load DataFrame from {parquet_file_path}")
            df = pd.read_parquet(parquet_file_path, filesystem=filesystem)
            logger.info("Loaded DataFrame from cache")

            return df, all_splits

        except FileNotFoundError:
            logger.warning("No cached data found, rebuilding data...")

    # If force_rebuild is True or cache is not found, process and store the data
    logger.info("Processing data and storing chunked documents and DataFrame")

    df, all_splits = preprocess_and_store_data(config, filesystem)
    return df, all_splits


# PATH BUILDER FOR S3 STORAGE ------------------------------------


def _s3_path_intermediate_collection(config: Mapping[str, Any]) -> str:
    """
    Build the intermediate storage path for chunked documents on S3.

    Parameters:
    - s3_bucket: str, the name of the S3 bucket.
    - chunk_overlap: int, the chunk overlap value (can be None).
    - chunk_size: int, the chunk size (can be None).
    - max_pages: int, the maximum number of pages (can be None).

    Returns:
    - str: The constructed S3 path for saving the chunked documents.
    """

    model_id = config.get('chunk_overlap', "OrdalieTech/Solon-embeddings-large-0.1")
    chunk_overlap = int(config['chunk_overlap']) if config.get('chunk_overlap') else None
    chunk_size = int(config['chunk_size']) if config.get('chunk_size') else None
    max_pages = int(config['max_pages']) if config.get('max_pages') else None
    return (
        f"s3://{config['s3_bucket']}/data/chunked_documents/"
        f"{model_id=}/"
        f"chunk_overlap={chunk_overlap}/chunk_size={chunk_size}/max_pages={max_pages}/"
        "docs.jsonl"
    )


# SAVE AND LOAD DOCUMENTS AS JSON ---------------------------------


def save_docs_to_jsonl(documents: t.Iterable[Document], file_path: str, fs: s3fs.S3FileSystem) -> None:
    """
    Save a list of Document objects to a JSONL file on S3 using s3fs.

    :param documents: Iterable of Document objects to be saved.
    :param file_path: The S3 path where the JSONL file will be saved (e.g., "s3://bucket-name/path/to/file.jsonl").
    :param fs: s3fs.S3FileSystem object for handling S3 file operations.
    """
    with fs.open(file_path, mode="w") as f:
        with jsonlines.Writer(f) as writer:
            for doc in documents:
                writer.write(doc.dict())  # Assuming Document has a .dict() method


def load_docs_from_jsonl(file_path: str, fs: s3fs.S3FileSystem) -> t.Iterable[Document]:
    """
    Load Document objects from a JSONL file on S3 using s3fs.

    :param file_path: The S3 path where the JSONL file is stored (e.g., "s3://bucket-name/path/to/file.jsonl").
    :param fs: s3fs.S3FileSystem object for handling S3 file operations.
    :return: Iterable of Document objects loaded from the JSONL file.
    """
    documents = []
    with fs.open(file_path, mode="r") as f:
        with jsonlines.Reader(f) as reader:
            for doc in reader:
                documents.append(Document(**doc))  # Assuming Document can be instantiated from a dict
    return documents
