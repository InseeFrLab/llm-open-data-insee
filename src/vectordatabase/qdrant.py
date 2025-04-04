import requests
import tempfile
import logging
import mlflow

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from langchain_openai import OpenAIEmbeddings


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


def create_client_and_collection_qdrant(
    url: str, api_key: str,
    collection_name: str = None,
    model_max_len: int = None,
    **kwargs
):
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


def get_number_docs_collection_qdrant(
    client: QdrantClient,
    collection_name: str,
):
    collection_info = client.get_collection(collection_name=collection_name)
    return collection_info.points_count


def create_collection_alias_qrant(
    client: QdrantClient,
    initial_collection_name: str,
    alias_collection_name: str
):
    client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(collection_name=initial_collection_name, alias_name=alias_collection_name)
                )
            ]
        )


def create_snapshot_collection_qdrant(
    client, collection_name, url, api_key
):
    snapshot = client.create_snapshot(collection_name=collection_name)
    url_snapshot = f"{url}/collections/{collection_name}/snapshots/{snapshot.name}"

    # Intermediate save snapshot in local for logging in MLFlow
    response = requests.get(url_snapshot, headers={"api-key": api_key}, timeout=60 * 10)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".snapshot") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name  # Store temp file path

    mlflow.log_artifact(local_path=temp_file_path)


def qdrant_vectorstore_as_retriever(
    client: QdrantVectorStore,
    collection_name: str,
    embedding_function: OpenAIEmbeddings,
    vector_name: str,
    number_retrieved_docs: int = 10
):

    db = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_function,
        vector_name=vector_name,
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": number_retrieved_docs}
    )

    return retriever
