import logging

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def _initialize_client_chroma(url: str, **kwargs) -> ClientAPI:
    logger.info("Setting connection")
    client = chromadb.HttpClient(host=url, port=443, ssl=True, settings=Settings())

    return client


def _initialize_collection_chroma(client: ClientAPI, collection_name: str):
    collection = client.get_or_create_collection(name=collection_name)

    return collection


def create_client_and_collection_chroma(
    url: str, collection_name: str = None, **kwargs):
    """
    Create and return a Qdrant client after initializing a vector collection.
    """

    client = _initialize_client_chroma(url)

    if collection_name is None:
        logger.debug("No collection_name provided, skipping that step")
        return client

    logger.info(f"Creating vector collection ({collection_name})")

    _initialize_collection_chroma(client=client, collection_name=collection_name)

    return client


def database_from_documents_chroma(
    documents, emb_model, client, collection_name: str, **kwargs
):
    """
    Embed documents and create a Qdrant vector store from them.
    """

    logger.info("Putting documents in vector database")

    db = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=emb_model,
    )

    _ = db.add_documents(documents=documents)

    return db


def get_number_docs_collection_chroma(
    client: ClientAPI,
    collection_name: str,
):
    collection_info = _initialize_collection_chroma(client, collection_name).count()
    return collection_info


def chroma_vectorstore_as_retriever(
    client: ClientAPI,
    collection_name: str,
    embedding_function: OpenAIEmbeddings,
    number_retrieved_docs: int = 10,
    **kwargs,
):
    db = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": number_retrieved_docs})

    return retriever
