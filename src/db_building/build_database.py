import gc
import os

import s3fs
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from loguru import logger

from src.config import Configurable, DefaultFullConfig, FullConfig
from src.model_building import cache_model_from_hf_hub

logger.add("./logging/logs.log")
load_dotenv()

# BUILD VECTOR DATABASE FROM COLLECTION -------------------------

s3_path = "s3://projet-llm-insee-open-data/data/raw_data/applishare_solr_joined.parquet"
DIRAG_INTERMEDIATE_PARQUET = "./data/raw/dirag.parquet"


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "OrdalieTech/Solon-embeddings-large-0.1")
URL_QDRANT = os.getenv("EMBEDDING_MODEL", None)
API_KEY_QDRANT = os.getenv("API_KEY_QDRANT", None)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "dirag_solon")
logger.debug(f"Using {EMBEDDING_MODEL} for database retrieval")
logger.debug(f"Setting {URL_QDRANT} as vector database endpoint")


@Configurable()
def build_vector_database(
    filesystem: s3fs.S3FileSystem,
    embedding_model: str | None = None,
    config: FullConfig = DefaultFullConfig(),
    return_none_on_fail=False,
    document_database: list[Document] | None = None,
) -> QdrantVectorStore | None:
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

    # LOADING EMBEDDING -----------------------

    logger.info(f"Loading embedding model: {embedding_model} on {config.emb_device}")

    cache_model_from_hf_hub(embedding_model, hf_token=os.environ.get("HF_TOKEN"))

    emb_model = HuggingFaceEmbeddings(  # load from sentence transformers
        model_name=embedding_model,
        model_kwargs={"device": config.emb_device},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        show_progress=False,
    )

    logger.info("Building the vector database from model and chunked docs")

    # Loop through the chunks and build the Chroma database
    try:
        db = QdrantVectorStore.from_documents(
            document_database,
            emb_model,
            url=URL_QDRANT,
            api_key=API_KEY_QDRANT,
            vector_name=EMBEDDING_MODEL,
            prefer_grpc=False,
            port="443",
            https="true",
            collection_name=COLLECTION_NAME,
        )
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


# LOAD RETRIEVER -------------------------------


@Configurable()
def load_retriever(
    vectorstore: QdrantVectorStore | None = None,
    retriever_params: dict | None = None,
    config: FullConfig = DefaultFullConfig(),
) -> tuple[VectorStoreRetriever, QdrantVectorStore]:
    if retriever_params is None:
        retriever_params = {"search_type": "similarity", "search_kwargs": {"k": 30}}

    search_kwargs = retriever_params.get("search_kwargs", {"k": 20})

    # Set up a retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    return retriever, vectorstore
