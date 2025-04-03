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
):
    total_docs = len(documents)

    if size_step is None:
        size_step = max(total_docs // 10, 1)
    if chunk_size is None:
        chunk_size = max(total_docs // 10, 1)

    logger.info(f"Number of documents to embed: {total_docs}")

    filtered_documents = _filter_valid_documents(documents, content_attr=content_attr, size_step=size_step)

    _embed_documents_in_chunks(filtered_documents, emb_model, collection_name, url, api_key, chunk_size)


def _filter_valid_documents(documents, content_attr: str, size_step: int):
    """
    Filters documents that have non-empty content in the given attribute.
    Logs progress every `size_step` documents.
    """
    total_docs = len(documents)
    logstep = 1 + (total_docs // size_step)
    filtered = []

    for i, doc in enumerate(documents):
        doc_id = doc.metadata.get("id", "unknown")
        doc_index = doc.metadata.get("index", i)

        if i % logstep == 0:
            logger.info(
                f"Filtering document {doc_index} (id={doc_id}) -- {i}/{total_docs} ({100 * i / total_docs:.2f}%)"
            )

        content = getattr(doc, content_attr, None)
        if not content:
            logger.warning(f"Skipping empty document at index {doc_index} (id={doc_id})")
            continue

        filtered.append(doc)

    logger.info(f"Filtered down to {len(filtered)} valid documents.")
    return filtered


def _embed_documents_in_chunks(documents, emb_model, collection_name: str, url: str, api_key: str, chunk_size: int):
    """
    Splits documents into chunks and sends each chunk to the vector store.
    """
    total_docs = len(documents)

    logger.info(f"Starting chunked ingestion with chunk size = {chunk_size}")

    for idx, batch_start in enumerate(range(0, total_docs, chunk_size), start=1):
        batch = documents[batch_start : batch_start + chunk_size]
        logger.info(
            f"Processing batch {idx}: docs {batch_start}–{batch_start + len(batch) - 1} "
            f"({100 * batch_start / total_docs:.2f}%)"
        )

        database_from_documents_qdrant(
            documents=batch, emb_model=emb_model, collection_name=collection_name, url=url, api_key=api_key
        )
