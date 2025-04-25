from loguru import logger

import s3fs
import pickle
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

    if chunking_args.get("strategy") is None:
        logger.info("Strategy is None, returning the initial documents")
        return documents_before_chunking

    if load_from_cache is True:

        documents_chunked = _get_cache_pickle(
            cache_file_documents=path_for_cache,
            filesystem=filesystem
        )

        logger.info("Documents have been retrieved from cache")

        if max_pages is not None:
            logger.debug(f"Limiting database to {max_pages} pages")
            unique_pages = set([docs.metadata['url'] for docs in documents])
            documents_chunked = [
                docs for docs in documents_chunked if docs.metadata['url'] in unique_pages
            ]
            data = data.head(max_pages)


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
            _write_cache_pickle(
                documents_chunked,
                cache_file_documents=path_for_cache,
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


def _get_cache_pickle(
    cache_file_documents: str, filesystem: s3fs.S3FileSystem
):

    if filesystem.lexists(cache_file_documents) is False:
        raise ValueError(f"No file found at {cache_file_documents}")

    with filesystem.open(cache_file_documents, "rb") as f:
        documents = pickle.load(f)

    return documents


def _write_cache_dataframe(
    df: pd.DataFrame, filesystem: s3fs.S3FileSystem, cache_file_documents: str
):

    df.to_parquet(cache_file_documents, filesystem=filesystem)
    logger.info(f"Dataset has been written at {cache_file_documents} location")


def _write_cache_pickle(
    data, filesystem: s3fs.S3FileSystem, cache_file_documents: str
):

    with filesystem.open(cache_file_documents, "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Dataset has been written at {cache_file_documents} location")

