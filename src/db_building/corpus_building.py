import logging
from typing import Tuple
import pandas as pd
import s3fs

import typing as t
import jsonlines
from langchain.schema import Document

from src.config import (
    CHROMA_DB_LOCAL_DIRECTORY,
    COLLECTION_NAME,
    EMB_DEVICE,
    EMB_MODEL_NAME,
    S3_BUCKET,
)

from .utils_db import parse_xmls
from .document_chunker import chunk_documents


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# PARAMETERS ---------------------------------------------------

DEFAULT_WEB4G_PATH = "data/raw_data/applishare_solr_joined.parquet"
DEFAULT_RMES_PATH = "data/raw_data/applishare_solr_joined.parquet"


DEFAULT_LOCATIONS = {
    "web4g_data": DEFAULT_WEB4G_PATH,
    "rmes_data": DEFAULT_RMES_PATH
}


# CHUNKING ALL DATASET ---------------------------------------------------


def preprocess_and_store_data(
    filesystem: s3fs.S3FileSystem,
    s3_bucket: str,
    location_dataset: dict = DEFAULT_LOCATIONS,
    **kwargs
) -> Tuple[pd.DataFrame, t.Iterable[Document]]:
    """
    Process data from S3, chunk the documents, and store them in an intermediate location.
    
    Parameters:
    - filesystem: s3fs.S3FileSystem object for interacting with S3.
    - s3_bucket: str, the name of the S3 bucket.
    - location_dataset: dict, paths to the main data files in the S3 bucket.
    - kwargs: Optional keyword arguments for data processing and chunking (e.g., 'max_pages', 'chunk_overlap', 'chunk_size').
    
    Returns:
    - Tuple containing the processed DataFrame and chunked documents (list of Document objects).
    """
    
    # Handle data loading, parsing, and splitting
    df, all_splits = _preprocess_data(
        filesystem=filesystem,
        s3_bucket=s3_bucket,
        location_dataset=location_dataset,
        **kwargs
    )

    logging.info("Saving chunked documents in an intermediate location")

    # Extract parameters from kwargs
    chunk_overlap = kwargs.get('chunk_overlap', None)
    chunk_size = kwargs.get('chunk_size', None)
    max_pages = kwargs.get('max_pages', None)

    # Create the storage path for intermediate data
    data_intermediate_storage = _s3_path_intermediate_collection(
        s3_bucket=s3_bucket,
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size,
        max_pages=max_pages
    )


    # Save chunked documents to JSONL using S3 and s3fs
    save_docs_to_jsonl(all_splits, data_intermediate_storage, filesystem)

    # Save the DataFrame as a Parquet file in the same directory
    parquet_file_path = data_intermediate_storage.replace("docs.jsonl", "corpus.parquet")
    df.loc[:, ~df.columns.isin(["dateDiffusion"])] = (
        df.loc[:, ~df.columns.isin(["dateDiffusion"])].astype(str)
    )    
    df.to_parquet(parquet_file_path, filesystem=filesystem, index=False)
    logging.info(f"DataFrame saved to {parquet_file_path}")

    return df, all_splits


def _preprocess_data(
    filesystem: s3fs.S3FileSystem,
    s3_bucket: str = S3_BUCKET,
    location_dataset: dict = DEFAULT_LOCATIONS,
    **kwargs
):
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

    # Defining data locations
    web4g_path = location_dataset.get("web4g_data", DEFAULT_WEB4G_PATH)
    data_path_rmes = location_dataset.get("rmes_data", DEFAULT_RMES_PATH)
    
    # Load main data from parquet file
    logging.info(f"Input data extracted from s3://{s3_bucket}")
    data = pd.read_parquet(f"s3://{s3_bucket}/{web4g_path}", filesystem=filesystem)

    # Limit the number of pages if specified
    if kwargs.get("max_pages") is not None:
        data = data.head(kwargs.get("max_pages"))

    # Parse the XML content
    parsed_pages = parse_xmls(data)

    # Merge parsed XML data with the original data
    df = data.set_index("id").merge(
        pd.DataFrame(parsed_pages), left_index=True, right_index=True
    )

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
    data_rmes = pd.read_parquet(f"s3://{s3_bucket}/{data_path_rmes}", filesystem=filesystem)

    # Concatenate the original data with the RMES data
    df = pd.concat([df, data_rmes])

    # Fill NaN values with empty strings (for compatibility with Chroma metadata)
    df = df.fillna(value="")

    # Chunk the documents (using tokenizer if specified in kwargs)
    all_splits = chunk_documents(data=df, **kwargs)

    return df, all_splits


# RECHUNKING OR LOADING FROM DATA STORE --------------------------

def build_or_use_from_cache(
    filesystem: s3fs.S3FileSystem,
    s3_bucket: str,
    location_dataset: dict = DEFAULT_LOCATIONS,
    force_rebuild: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, t.Iterable[Document]]:
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

    # Extract parameters from kwargs
    chunk_overlap = kwargs.get('chunk_overlap', None)
    chunk_size = kwargs.get('chunk_size', None)
    max_pages = kwargs.get('max_pages', None)

    # Create the storage path for intermediate data
    data_intermediate_storage = _s3_path_intermediate_collection(
        s3_bucket=s3_bucket,
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size,
        max_pages=max_pages
    )

    # Create the Parquet file path (corpus.parquet)
    parquet_file_path = data_intermediate_storage.replace("docs.jsonl", "corpus.parquet")

    # Check if we should use the cached data
    if not force_rebuild:
        try:
            # Attempt to load the cached documents from S3
            logging.info(f"Attempting to load chunked documents from {data_intermediate_storage}")
            all_splits = load_docs_from_jsonl(data_intermediate_storage, filesystem)
            logging.info("Loaded chunked documents from cache")
            
            # Attempt to load the cached DataFrame from S3
            logging.info(f"Attempting to load DataFrame from {parquet_file_path}")
            df = pd.read_parquet(parquet_file_path, filesystem=filesystem)
            logging.info("Loaded DataFrame from cache")

            return df, all_splits

        except FileNotFoundError:
            logging.warning("No cached data found, rebuilding data...")

    # If force_rebuild is True or cache is not found, process and store the data
    logging.info("Processing data and storing chunked documents and DataFrame")
    df, all_splits = preprocess_and_store_data(
        filesystem=filesystem,
        s3_bucket=s3_bucket,
        location_dataset=location_dataset,
        **kwargs
    )

    return df, all_splits



# PATH BUILDER FOR S3 STORAGE ------------------------------------


def _s3_path_intermediate_collection(
    s3_bucket: str = S3_BUCKET,
    chunk_overlap: int = None,
    chunk_size: int = None,
    max_pages: int = None
) -> str:
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
    return (
        f"s3://{s3_bucket}/data/chunked_documents/"
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