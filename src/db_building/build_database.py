import gc
import os

import pandas as pd
import s3fs
from chromadb.config import Settings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.config import Configurable, DefaultFullConfig, FullConfig
from src.model_building import cache_model_from_hf_hub

from .corpus_building import build_or_load_document_database
from .utils_db import split_list

logger.add("./logging/logs.log")


# BUILD VECTOR DATABASE FROM COLLECTION -------------------------


@Configurable()
def build_vector_database(
    filesystem: s3fs.S3FileSystem,
    embedding_model: str | None = None,
    config: FullConfig = DefaultFullConfig(),
    return_none_on_fail=False,
    document_database: tuple[pd.DataFrame, list[Document]] | None = None,
) -> Chroma | None:
    """
    Build vector database from documents database

    Args:
    filesystem: The filesystem object for interacting with S3.
    config: Keyword arguments for building the vector database:
        - emb_device (str): device to run the embedding model on
        - embedding_model (str): the embedding model to use
        - collection_name (str): langchain collection name
        - chroma_db_local_path (str): local path to store the database
        - batch_size_embedding (int): batch size for embedding
    document_database: the document database. Will be build_or_loaded if unspecified.

    Returns:
    The built Chroma vector database
    """
    logger.info("Building the vector database from documents")

    if embedding_model is None:
        embedding_model = config.embedding_model

    # Call the process_data function to handle data loading, parsing, and splitting
    df, all_splits = document_database or build_or_load_document_database(filesystem, config)

    logger.info(f"Loading embedding model: {embedding_model} on {config.emb_device}")

    cache_model_from_hf_hub(embedding_model, hf_token=os.environ.get("HF_TOKEN"))

    emb_model = HuggingFaceEmbeddings(  # load from sentence transformers
        model_name=embedding_model,
        model_kwargs={"device": config.emb_device},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        show_progress=False,
    )

    logger.info("Building the Chroma vector database from model and chunked docs")
    logger.info(f"The database will temporarily be stored in {config.chroma_db_local_path}")

    split_docs_chunked = split_list(all_splits, config.batch_size_embedding)
    nb_chunks = 1 + (len(all_splits) - 1) // config.batch_size_embedding

    # Loop through the chunks and build the Chroma database
    try:
        db = Chroma(
            collection_name=config.collection_name,
            persist_directory=config.chroma_db_local_path,
            embedding_function=emb_model,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        logger.info(f"Empty new database created. Adding {len(all_splits)} documents...")
        for chunk_count, split_docs_chunk in enumerate(split_docs_chunked):
            logger.info(
                f"Max len in chunk: {max([len(doc.page_content) for doc in list(split_docs_chunk)])}"
            )  # Just to check for memory issues
            db.add_documents(list(split_docs_chunk))
            ratio_docs_processed = min(1.0, config.batch_size_embedding * (chunk_count + 1) / len(all_splits))
            logger.info(f"Chunk: {chunk_count+1}/{nb_chunks} ({100*ratio_docs_processed:.0f}%)")
    except Exception as e:
        logger.error(f"An error occurred while building the Chroma database: {e}")
        if return_none_on_fail:
            return None
        else:
            raise

    # Cleanup after successful execution
    del emb_model
    gc.collect()

    logger.success("Vector database built successfully!")
    return db


# LOAD VECTOR DATABASE FROM LOCAL DIRECTORY --------------------


@Configurable()
def load_vector_database_from_local(
    persist_directory: str | None = None, embedding_model: str | None = None, config: FullConfig = DefaultFullConfig()
) -> Chroma:
    """
    Load Chroma vector database from local directory.

    Assumes, without checking, that the embedding function matches `config.embedding_model`

    Args:
    persist_directory: path to the directory from. If empty, `config.chroma_db_local_path` is used
    config: configuration

    Returns:
    The loaded Chroma vector database
    """

    logger.info("Loading Chroma vector database from local session")

    if persist_directory is None:
        persist_directory = config.chroma_db_local_path

    if embedding_model is None:
        embedding_model = config.embedding_model

    emb_model = HuggingFaceEmbeddings(
        model_name=embedding_model,
        multi_process=False,
        model_kwargs={"device": config.emb_device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )
    db = Chroma(
        collection_name=config.collection_name,
        persist_directory=persist_directory,
        embedding_function=emb_model,
    )
    logger.info(f"Database (collection {config.collection_name}) reloaded from directory {persist_directory}")
    return db


# LOAD RETRIEVER -------------------------------


@Configurable()
def load_retriever(
    vectorstore: Chroma | None = None, retriever_params: dict | None = None, config: FullConfig = DefaultFullConfig()
) -> tuple[VectorStoreRetriever, Chroma]:
    if vectorstore is None:
        vectorstore = load_vector_database_from_local(persist_directory=None, config=config)
    if retriever_params is None:
        retriever_params = {"search_type": "similarity", "search_kwargs": {"k": 30}}

    search_kwargs = retriever_params.get("search_kwargs", {"k": 20})

    # Set up a retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    return retriever, vectorstore
