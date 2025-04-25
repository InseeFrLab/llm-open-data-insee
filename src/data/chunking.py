import logging

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document

logger = logging.getLogger(__name__)

# RECURSIVE_HEADERS_TO_CHUNK = ["#", "##", "###", "####", "#####", "######"]
RECURSIVE_HEADERS_TO_CHUNK = ["\n\n", "\n"]


def chunk_documents(
    documents: list[Document],
    strategy: str = "recursive",
    separators=RECURSIVE_HEADERS_TO_CHUNK,
    minimal_size_documents=500,
    **kwargs,
) -> list[Document]:
    logging.info("Building the list of document objects")
    logging.info(f"The following parameters have been applied: {kwargs}")

    if strategy is None or strategy == "None":
        docs_processed = documents

    # Initialize token splitter
    if strategy.lower() == "recursive":
        docs_processed = RecursiveCharacterTextSplitter(separators=separators, **kwargs).split_documents(
            [documents[-1]]
        )
    elif strategy.lower() == "character":
        docs_processed = CharacterTextSplitter(
            separator=" ", length_function=len, is_separator_regex=False, **kwargs
        ).create_documents([d.page_content for d in documents], metadatas=[d.metadata for d in documents])

    logging.info(f"Number of created chunks: {len(docs_processed)} in the Vector Database")
    if minimal_size_documents is not None:
        logging.info(f"Keeping only documents with more than {minimal_size_documents} characters")
        docs_processed = [docs for docs in docs_processed if len(docs.page_content) > minimal_size_documents]

    return docs_processed
