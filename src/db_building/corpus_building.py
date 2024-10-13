import logging
import pandas as pd
import s3fs

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


def process_data(
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

    return all_splits
