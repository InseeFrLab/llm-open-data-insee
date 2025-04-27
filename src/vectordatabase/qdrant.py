
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# logger = logging.getLogger(__name__)


def _initialize_client_qdrant(url: str, api_key: str) -> QdrantClient:
    logger.info("Setting connection")
    client = QdrantClient(url=url, api_key=api_key, port="443", https=True)

    return client


def _initialize_collection_qdrant(client: QdrantClient, collection_name: str, model_max_len: float, vector_name: str):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            vector_name: VectorParams(size=model_max_len, distance=Distance.COSINE),
        },
    )
    return client


def create_client_and_collection_qdrant(
    url: str, api_key: str, vector_name: str, collection_name: str = None, model_max_len: int = None, **kwargs
):
    """
    Create and return a Qdrant client after initializing a vector collection.
    """

    client = _initialize_client_qdrant(url, api_key)

    if collection_name is None:
        logger.debug("No collection_name provided, skipping that step")
        return client

    if client.collection_exists(collection_name=collection_name) is True:
        logger.debug(f"Collection {collection_name} exists, skipping creation")
        return client

    logger.info(f"Creating vector collection ({collection_name})")

    _initialize_collection_qdrant(
        client=client, collection_name=collection_name, vector_name=vector_name, model_max_len=model_max_len
    )

    return client


def database_from_documents_qdrant(documents, emb_model, client, collection_name: str, **kwargs):
    """
    Embed documents and create a Qdrant vector store from them.
    """

    logger.info("Putting documents in vector database")

    db = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=emb_model, vector_name=emb_model.model
    )

    _ = db.add_documents(documents=documents)

    return db


def get_number_docs_collection_qdrant(
    client: QdrantClient,
    collection_name: str,
):
    collection_info = client.get_collection(collection_name=collection_name)
    return collection_info.points_count


def qdrant_vectorstore_as_retriever(
    client: QdrantVectorStore,
    collection_name: str,
    embedding_function: OpenAIEmbeddings,
    vector_name: str,
    number_retrieved_docs: int = 10,
):
    db = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_function,
        vector_name=vector_name,
    )

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": number_retrieved_docs})

    return retriever
