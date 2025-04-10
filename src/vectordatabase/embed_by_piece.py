from chromadb.api import ClientAPI
from loguru import logger

from .qdrant import database_from_documents_qdrant


def chunk_documents_and_store(
    documents,
    emb_model,
    collection_name: str,
    url: str,
    api_key: str,
    chunk_size: int = None,
    content_attr: str = "page_content",
    size_step: int = None,
    engine: str = "qdrant",
    client: ClientAPI = None,
):
    total_docs = len(documents)

    if size_step is None:
        size_step = max(total_docs // 10, 1)
    if chunk_size is None:
        chunk_size = max(total_docs // 10, 1)

    logger.info(f"Number of documents to embed: {total_docs}")

    filtered_documents = _filter_valid_documents(documents, content_attr=content_attr, size_step=size_step)

    _embed_documents_in_chunks(
        filtered_documents, emb_model, collection_name, url, api_key, chunk_size, engine=engine, client=client
    )


def _filter_valid_documents(documents, content_attr: str, size_step: int):
    """
    Filters documents that have non-empty content in the given attribute.
    Logs progress every `size_step` documents.
    """
    filtered = []

    for i, doc in enumerate(documents):
        doc_id = doc.metadata.get("id", "unknown")
        doc_index = doc.metadata.get("index", i)

        content = getattr(doc, content_attr, None)
        if not content:
            logger.warning(f"Skipping empty document at index {doc_index} (id={doc_id})")
            continue

        filtered.append(doc)

    return filtered


def _embed_documents_in_chunks(
    documents,
    emb_model,
    collection_name: str,
    url: str,
    api_key: str,
    chunk_size: int,
    engine="qdrant",
    client=None,
):
    """
    Splits documents into chunks and sends each chunk to the vector store.
    """

    if engine not in ["qdrant", "chroma"]:
        raise ValueError("Only ChromaDB or Qdrant engines for database management are supported")
    if engine == "chroma" and client is None:
        raise ValueError("client is optional for qdrant engine but mandatory for chroma")

    total_docs = len(documents)

    args_database_constructor = {"emb_model": emb_model, "collection_name": collection_name, "client": client}

    logger.info(f"Starting chunked ingestion with chunk size = {chunk_size}")

    for idx, batch_start in enumerate(range(0, total_docs, chunk_size), start=1):
        batch = documents[batch_start : (batch_start + chunk_size)]
        logger.info(
            f"Processing batch {idx}: docs {batch_start}–{batch_start + len(batch) - 1} "
            f"({100 * batch_start / total_docs:.2f}%)"
        )

        database_construction_func = database_from_documents_qdrant
        if engine == "chroma":
            database_construction_func = database_from_documents_qdrant

        database_construction_func(documents=batch, **args_database_constructor)
