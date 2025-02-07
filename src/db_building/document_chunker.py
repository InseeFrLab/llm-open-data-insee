import logging

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from transformers import AutoTokenizer

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]


def chunk_documents(
    data: pd.DataFrame,
    embedding_model: str,
    markdown_split: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[Document]:
    """
    Chunks documents from a dataframe into smaller pieces using specified tokenizer settings
    or custom settings.

    Parameters:
    - data (pd.DataFrame): The dataframe containing documents to be chunked.
    - embedding_model (str): The name of the Hugging Face tokenizer to use.
    - markdown_split (bool): Whether to split markdown headers into separate chunks.
    - chunk_size (int, optional): Size of each chunk if not using hf_tokenizer.
    - chunk_overlap (int, optional): Overlap size between chunks if not using hf_tokenizer.
    - separators (list, optional): List of separators to use for splitting the text.

    Returns:
    - list[Document]: The list of processed unique document chunks
    """

    logging.info("Building the list of document objects")

    # advantage of using a loader
    # No need to know which metadata are stored in the dataframe
    # Every column except page_content_column contains metadata
    document_list = DataFrameLoader(data, page_content_column="content").load()

    if markdown_split:
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)
        document_list = make_md_splits(document_list, markdown_splitter)

    # Load the tokenizer
    autokenizer = AutoTokenizer.from_pretrained(embedding_model)

    if autokenizer.model_max_length is not None:
        chunk_size = autokenizer.model_max_length

    # Initialize token splitter
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        autokenizer,
        chunk_size=chunk_size,
        # chunk_overlap=chunk_overlap,
        separators=separators,
    )

    # Split documents into chunks
    docs_processed = text_splitter.split_documents(document_list)

    # Remove duplicates
    unique_texts = set()
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts.add(doc.page_content)
            docs_processed_unique.append(doc)

    logging.info(f"Number of created chunks: {len(docs_processed_unique)} in the Vector Database")

    return docs_processed_unique


def make_md_splits(document_list: list[Document], markdown_splitter: MarkdownHeaderTextSplitter) -> list[Document]:
    """
    Splits the content of each document in the document list based on Markdown headers,
    and preserves the original metadata in each split section.

    Args:
        document_list (list[Document]): List of Document objects to be split.
        markdown_splitter (MarkdownHeaderTextSplitter): An instance of MarkdownHeaderTextSplitter
            used to perform the text splitting based on Markdown headers.

    Returns:
        list[Document]: List of split Document objects with updated metadata.
    """
    splitted_docs = []

    for doc in document_list:
        # Split the document content into sections based on Markdown headers
        md_header_splits = markdown_splitter.split_text(doc.page_content)

        for md_section in md_header_splits:
            # Update metadata for each split section with the original document's metadata
            md_section.metadata.update(doc.metadata)
            splitted_docs.append(md_section)

    return splitted_docs
