from collections.abc import Iterable
import jsonlines

from loguru import logger

import s3fs
import pandas as pd

from langchain_core.documents.base import Document

from .corpus import constructor_corpus
from .parsing import parse_documents
from .chunking import chunk_documents


def parse_documents_or_load_from_cache(
    path_for_cache: str,
    load_from_cache: bool = False,
    max_pages: int = None,
    filesystem: s3fs = None,
    corpus_constructor_args: dict = None,
    force_rebuild: bool = False
):

    if corpus_constructor_args is None:
        corpus_constructor_args = {}

    if force_rebuild is True:
        load_from_cache = False

    if filesystem.lexists(path_for_cache) is False and load_from_cache is True:
        logger.warning(
            f"File does not exist in cache ({path_for_cache}), data will be reconstructed from scratch"
        )
        load_from_cache = False


    if load_from_cache is True:

        data = _get_cache_dataframe(
            cache_file_documents=path_for_cache,
            filesystem=filesystem
        )

        logger.info("Data have been retrieved from cache")

        if max_pages is not None:
            logger.debug(f"Limiting database to {max_pages} pages")
            data = data.head(max_pages)

    else:

        data = constructor_corpus(
            **corpus_constructor_args
        )

        if max_pages is not None:
            logger.debug(f"Limiting database to {max_pages} pages")
            data = data.head(max_pages)

        data = parse_documents(data)

        if max_pages is None:
            # Cache this dataset
            _write_cache_dataframe(
                data,
                cache_file_documents=path_for_cache,
                filesystem=filesystem
            )


    return data


def chunk_documents_or_load_from_cache(
    documents_before_chunking: list[Document],
    path_for_cache: str,
    load_from_cache: bool = False,
    max_pages: int = None,
    filesystem: s3fs = None,
    force_rebuild: bool = False,
    chunking_args: dict = None
):

    if chunking_args is None:
        chunking_args = {}

    if force_rebuild is True:
        load_from_cache = False

    if filesystem.lexists(path_for_cache) is False and load_from_cache is True:
        logger.warning(
            f"File does not exist in cache ({path_for_cache}), data will be reconstructed from scratch"
        )
        load_from_cache = False

    if chunking_args.get("strategy") == 'None':
        logger.info("Strategy is None, returning the initial documents")
        return documents_before_chunking

    if load_from_cache is True:

        documents_chunked = _get_cache_jsonl(
            file_path=path_for_cache,
            filesystem=filesystem
        )

        logger.info("Documents have been retrieved from cache")

        if max_pages is not None:
            logger.debug(f"Limiting database to {max_pages} pages")
            unique_pages = set([docs.metadata['url'] for docs in documents_before_chunking])
            documents_chunked = [
                docs for docs in documents_chunked if docs.metadata['url'] in unique_pages
            ]


    else:
        documents_chunked = chunk_documents(
            documents_before_chunking,
            **chunking_args
        )
        unique_pages = set([docs.metadata['url'] for docs in documents_before_chunking])
        documents_chunked = [
                docs for docs in documents_chunked if docs.metadata['url'] in unique_pages
        ]

        if max_pages is None:
            # Cache this dataset
            _write_cache_jsonl(
                documents_chunked,
                file_path=path_for_cache,
                filesystem=filesystem
            )


    return documents_chunked



# UTILITIES --------------------------------------

def _get_cache_dataframe(
    cache_file_documents: str, filesystem: s3fs.S3FileSystem
):

    if filesystem.lexists(cache_file_documents) is False:
        raise ValueError(f"No file found at {cache_file_documents}")

    return pd.read_parquet(cache_file_documents, filesystem=filesystem)


def _write_cache_dataframe(
    df: pd.DataFrame, filesystem: s3fs.S3FileSystem, cache_file_documents: str
):

    df.to_parquet(cache_file_documents, filesystem=filesystem)
    logger.info(f"Dataset has been written at {cache_file_documents} location")


def _get_cache_jsonl(file_path: str, filesystem: s3fs.S3FileSystem) -> list[Document]:
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


def _write_cache_jsonl(documents: Iterable[Document], file_path: str, filesystem: s3fs.S3FileSystem) -> None:
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



def _write_cache_pickle(
    data, filesystem: s3fs.S3FileSystem, cache_file_documents: str
):

    with filesystem.open(cache_file_documents, "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Dataset has been written at {cache_file_documents} location")

