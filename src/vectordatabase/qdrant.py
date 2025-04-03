import logging

from langchain.vectorstores import Qdrant as QdrantVectorStore
from langchain.embeddings import OpenAIEmbeddings


from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger(__name__)


def _initialize_client_qdrant(url: str, api_key: str) -> QdrantClient:
    logger.info("Setting connection")
    client = QdrantClient(url=url, api_key=api_key, port="443", https=True)

    return client


def _initialize_collection_qdrant(client: QdrantClient, collection_name: str, model_max_len: float):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=model_max_len, distance=Distance.COSINE),
    )
    return client


def create_client_and_collection_qdrant(url: str, api_key: str, collection_name: str = None, model_max_len: int = None):
    """
    Create and return a Qdrant client after initializing a vector collection.
    """

    client = _initialize_client_qdrant(url, api_key)

    if collection_name is None:
        logger.debug("No collection_name provided, skipping that step")
        return client

    logger.info(f"Creating vector collection ({collection_name})")

    _initialize_collection_qdrant(client=client, collection_name=collection_name, model_max_len=model_max_len)

    return client


def database_from_documents_qdrant(
    documents,
    emb_model,
    collection_name: str,
    url: str,
    api_key: str,
):
    """
    Embed documents and create a Qdrant vector store from them.
    """

    logger.info("Putting documents in vector database")

    db = QdrantVectorStore.from_documents(
        documents,
        emb_model,
        url=url,
        api_key=api_key,
        vector_name=emb_model.model,
        prefer_grpc=False,
        port="443",
        https="true",
        collection_name=collection_name,
        force_recreate=True,
    )

    return db


def get_number_docs_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
):
    collection_info = qdrant_client.get_collection(collection_name=collection_name)
    return collection_info.points_count
