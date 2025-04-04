import logging
import chromadb

logger = logging.getLogger(__name__)

def _initialize_client_chroma(url: str, api_key: str):
    logger.info("Setting connection")
    client = QdrantClient(url=url, api_key=api_key, port="443", https=True)

    return client
