from collections.abc import Iterable

import jsonlines
import pandas as pd
import s3fs
from langchain.schema import Document
from loguru import logger

from src.config import Configurable, DefaultFullConfig, FullConfig

from .document_chunker import chunk_documents
from .utils_db import parse_xmls

logger.add("./logging/logs.log")

# CHUNKING ALL DATASET ---------------------------------------------------


@Configurable()
def build_document_database(
    filesystem: s3fs.S3FileSystem, config: FullConfig = DefaultFullConfig()
) -> tuple[pd.DataFrame, list[Document]]:
    """
    Process data from S3, chunk the documents, and store them in an intermediate location.

    Parameters:
    - config (Mapping[str, Any]): main run configuration. The following variables are accessed:
        - s3_bucket: the name of the S3 bucket.
        - chunk_overlap (int): the chunk overlap value (can be None).
        - chunk_size (int): the chunk size (can be None).
        - max_pages (int): the maximum number of pages (can be None).
        - rawdata_web4g_uri
        - rawdata_rmes_uri
        - embedding_model
        - markdown_split
        - separators
    - filesystem (s3fs.S3FileSystem): object for interacting with S3

    Returns:
    - Tuple containing the processed DataFrame and chunked documents (list of Document objects).
    """
    # Handle data loading, parsing, and splitting
    logger.info("Processing input data and storing chunked documents and DataFrame")
    df, all_splits = _preprocess_data(filesystem, config)

    logger.info(f"Saving chunked documents in JSONL format to {config.documents_jsonl_s3_path}")
    save_docs_to_jsonl(all_splits, config.documents_jsonl_s3_path, filesystem)

    # Save the DataFrame as a Parquet file in the same directory
    logger.info(f"Saving DataFrame as a Parquet file to {config.documents_parquet_s3_path}")
    df.loc[:, ~df.columns.isin(["dateDiffusion"])] = df.loc[:, ~df.columns.isin(["dateDiffusion"])].astype(str)
    df.to_parquet(config.documents_parquet_s3_path, filesystem=filesystem, index=False)

    logger.info("Document chunking successful.")
    return df, all_splits


@Configurable()
def _preprocess_data(
    filesystem: s3fs.S3FileSystem,
    config: FullConfig = DefaultFullConfig(),
    data: pd.DataFrame = None,
    embedding_model: str | None = None,
    skip_chunking: bool = False,
) -> tuple[pd.DataFrame, list[Document]]:
    """
    Process and merge data from multiple parquet sources, parse XML content,
    and split the documents into chunks for embedding.

    Parameters:
    - filesystem (s3fs.S3FileSystem): connexion object for interacting with S3
    - config (Mapping[str, Any]): main run configuration. The following variables are accessed:
        - max_pages
        - rawdata_web4g_uri: link to the web4g data
        - rawdata_rmes_uri: link to the rmes data
        - embedding_model: The name of the Hugging Face tokenizer to use.
        - markdown_split (bool): Whether to split markdown headers into separate chunks.
        - hf_tokenizer_name (str, optional): Name of the Hugging Face tokenizer to use.
        - chunk_size (int, optional): Size of each chunk if not using hf_tokenizer.
        - chunk_overlap (int, optional): Overlap size between chunks if not using hf_tokenizer.
        - separators (list, optional): List of separators to use for splitting the text.

    Returns:
    - all_splits: list or DataFrame containing the processed and chunked documents.
    """

    if embedding_model is None:
        embedding_model = config.embedding_model

    # Load main data from parquet file
    logger.info(f"Reading web4g data from {config.rawdata_web4g_uri}")

    if data is None:
        data = pd.read_parquet(config.rawdata_web4g_uri, filesystem=filesystem)
        # Load additional RMES data
        logger.info(f"Reading rmes data from {config.rawdata_rmes_uri}")
        data_rmes = pd.read_parquet(config.rawdata_rmes_uri, filesystem=filesystem)

    # Limit the number of pages if specified
    if config.max_pages is not None:
        data = data.head(config.max_pages)

    # Parse the XML content
    logger.info("Parsing XML content")
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

    # Concatenate the original data with the RMES data (if max_pages is not None)
    if config.max_pages is not None:
        df = pd.concat([df, data_rmes])

    # Fill NaN values with empty strings (for compatibility with Chroma metadata)
    df = df.fillna(value="")

    if skip_chunking is True:
        logger.success("Chunking is skipped, preprocessed is done")
        return df

    logger.info("Chunking corpus")

    # Chunk the documents (using tokenizer if specified in kwargs)
    all_splits = chunk_documents(
        data=df,
        embedding_model=embedding_model,
        markdown_split=config.markdown_split,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
    )

    logger.info("Corpus chunking has been successful")

    return df, all_splits


# RECHUNKING OR LOADING FROM DATA STORE --------------------------


@Configurable()
def load_document_database(
    filesystem: s3fs.S3FileSystem, config: FullConfig = DefaultFullConfig()
) -> tuple[pd.DataFrame, list[Document]]:
    """
    Load the document database from local cache.

    Args:
    filesystem: object for interacting with S3
    config: main run configuration. The following variables are accessed:

    Returns:
    Tuple containing the processed DataFrame and chunked documents (list of Document objects).

    Raises:
    FileNotFoundError: In case no database could be found in local cache
    """
    # Attempt to load the cached documents from S3
    logger.info(f"Attempting to load chunked documents from {config.documents_jsonl_s3_path}")
    all_splits = load_docs_from_jsonl(config.documents_jsonl_s3_path, filesystem)
    logger.info("Loaded chunked documents from cache")

    # Attempt to load the cached DataFrame from S3
    logger.info(f"Attempting to load DataFrame from {config.documents_parquet_s3_path}")
    df = pd.read_parquet(config.documents_parquet_s3_path, filesystem=filesystem)
    logger.info("Loaded DataFrame from cache")

    return df, all_splits


@Configurable()
def build_or_load_document_database(
    filesystem: s3fs.S3FileSystem, config: FullConfig = DefaultFullConfig()
) -> tuple[pd.DataFrame, list[Document]]:
    """
    Load the document database from local cache
    or download all documents from S3, process them and store the result in local cache.

    Args:
    filesystem: object for interacting with S3
    config: main run configuration. The following variables are accessed:
        - s3_bucket: the name of the S3 bucket.
        - force_rebuild (bool): force data processing and storage even if cache exists.
        - max_pages
        - chunk_overlap (int): the chunk overlap value (can be None).
        - chunk_size (int): the chunk size (can be None).
        - max_pages (int): the maximum number of pages (can be None).

    Returns:
    - Tuple containing the processed DataFrame and chunked documents (list of Document objects).
    """
    logger.info("Generating dataframe of chunked documents")
    logger.info(f"The document database will temporarily be stored in {config.data_dir_path}")

    # Check if we should use the cached data
    if not config.force_rebuild:
        try:
            return load_document_database(filesystem, config=config)
        except FileNotFoundError:
            logger.warning("No cached data found, rebuilding data...")

    # If force_rebuild is True or cache is not found, process and store the data
    logger.info("Processing input data and storing chunked documents and DataFrame")
    return build_document_database(filesystem, config)


# SAVE AND LOAD DOCUMENTS AS JSON ---------------------------------


def save_docs_to_jsonl(documents: Iterable[Document], file_path: str, filesystem: s3fs.S3FileSystem) -> None:
    """
    Save a list of Document objects to a JSONL file on S3 using s3fs.

    Args:
    documents: Document objects to be saved
    file_path: the S3 path where the JSONL file will be saved
      (e.g., "s3://bucket-name/path/to/file.jsonl").
    filesystem: s3fs.S3FileSystem object for handling S3 file operations.
    """
    with filesystem.open(file_path, mode="w") as f, jsonlines.Writer(f) as writer:
        for doc in documents:
            writer.write(doc.dict())  # Assuming Document has a .dict() method


def load_docs_from_jsonl(file_path: str, filesystem: s3fs.S3FileSystem) -> list[Document]:
    """
    Load Document objects from a JSONL file on S3 using s3fs.

    Args:
    file_path: The S3 path where the JSONL file is stored
        (e.g., "s3://bucket-name/path/to/file.jsonl").
    filesystem: s3fs.S3FileSystem object for handling S3 file operations.

    Returns:
    List of Document objects loaded from the JSONL file.
    """
    documents = []
    with filesystem.open(file_path, mode="r") as f, jsonlines.Reader(f) as reader:
        for doc in reader:
            documents.append(Document(**doc))
            # Assuming Document can be instantiated from a dict
    return documents
