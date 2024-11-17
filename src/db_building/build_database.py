import gc
import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd
import s3fs
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import RAGConfig

from .corpus_building import build_or_use_from_cache
from .utils_db import split_list

logger = logging.getLogger(__name__)


def parse_collection_name(collection_name: str) -> dict[str, str | int] | None:
    """
    Parse a concatenated string to extract the embedding model name, chunk size, and overlap size.
    :param concatenated_string: A string in the format 'embeddingmodelname_chunkSize_overlapSize'
    :return: A dictionary with the parsed values
    """
    try:
        # Split the string by the underscore delimiter
        parts = collection_name.split("_")

        # Ensure there are exactly three parts
        if len(parts) != 3:
            raise ValueError("String format is incorrect." "Expected format: 'modelname_chunkSize_overlapSize'")

        # Extract and assign the parts
        model_name = parts[0]
        chunk_size = int(parts[1])
        overlap_size = int(parts[2])

        # Return the parsed values in a dictionary
        return {
            "model_name": model_name,
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
        }
    except Exception as e:
        logger.error(f"Error parsing string: {e}")
        return None


# BUILD VECTOR DATABASE FROM COLLECTION -------------------------


def build_vector_database(
    filesystem: s3fs.S3FileSystem, config: Mapping[str, Any] = vars(RAGConfig())
) -> tuple[Chroma | None, pd.DataFrame]:
    """
    Build vector database from documents.

    Parameters:
    filesystem: s3fs.S3FileSystem
        The filesystem object for interacting with S3.
    config: Mapping[str, Any]
        Keyword arguments for building the vector database:
        - emb_device (str):
        - emb_model (str):
        - collection_name (str): langchain collection name
        - chroma_db_local_path (str):
    """
    logger.info(f"The database will temporarily be stored in {config['chroma_db_local_path']}")

    # Building embedding model using parameters from kwargs
    embedding_device = config["emb_device"]
    embedding_model = config["emb_model"]

    logger.info("Start building the database")

    # Call the process_data function to handle data loading, parsing, and splitting
    df, all_splits = build_or_use_from_cache(filesystem, config)

    logger.info("Document chunking is over, starting to embed them")

    logger.info("Loading embedding model")

    emb_model = HuggingFaceEmbeddings(  # load from sentence transformers
        model_name=embedding_model,
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        show_progress=False,
    )

    logger.info(f"Building embedding model: {embedding_model} on {embedding_device}")

    max_batch_size = 41600
    split_docs_chunked = split_list(all_splits, max_batch_size)
    nb_chunks = (len(all_splits) + 1) // max_batch_size

    # Loop through the chunks and build the Chroma database
    try:
        for chunk_count, split_docs_chunk in enumerate(split_docs_chunked):
            db = Chroma.from_documents(
                collection_name=config["collection_name"],
                documents=list(split_docs_chunk),
                persist_directory=config["chroma_db_local_path"],
                embedding=emb_model,
                client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
            )
            ratio_docs_processed = min(1.0, max_batch_size * chunk_count / len(all_splits))
            logger.info(f"Chunk: {chunk_count}/{nb_chunks} ({100*ratio_docs_processed:.0f}%)")

    except Exception as e:
        logger.error(f"An error occurred while building the Chroma database: {e}")

        # Returns None along with the dataframe in case of failure
        return None, df

    # Cleanup after successful execution
    del emb_model
    gc.collect()

    logger.info("The database has been built")
    return db, df


# RELOAD VECTOR DATABASE FROM DIRECTORY -------------------------


def reload_database_from_local_dir(
    persist_directory: str | None = None, config: Mapping[str, Any] = vars(RAGConfig())
) -> Chroma:
    """
    Reload Chroma vector database from local directory.
    """
    if persist_directory is None:
        persist_directory = config["chroma_db_local_path"]
    emb_model = HuggingFaceEmbeddings(
        model_name=config["emb_model"],
        multi_process=False,
        model_kwargs={"device": config["emb_device"], "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )
    db = Chroma(
        collection_name=config["collection_name"],
        persist_directory=persist_directory,
        embedding_function=emb_model,
    )
    logger.info(f"Database (collection {config['collection_name']}) reloaded from directory {persist_directory}")
    return db


# LOAD RETRIEVER -------------------------------


def load_retriever(
    vectorstore: Chroma | None = None,
    retriever_params: dict | None = None,
    config: Mapping[str, Any] = vars(RAGConfig()),
) -> tuple[VectorStoreRetriever, Chroma]:
    # Load vector database
    if vectorstore is None:
        logger.info("Reloading database in session")

        vectorstore = reload_database_from_local_dir(persist_directory=None, config=config)
    else:
        logger.info("vectorstore being provided, skipping the reloading")

    if retriever_params is None:
        retriever_params = {"search_type": "similarity", "search_kwargs": {"k": 30}}

    search_kwargs = retriever_params.get("search_kwargs", {"k": 20})

    # Set up a retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    return retriever, vectorstore
