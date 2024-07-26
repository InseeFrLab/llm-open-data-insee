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
    **kwargs,
) -> tuple[list[Document], dict]:
    """
    Chunks documents from a dataframe into smaller pieces using specified tokenizer settings or custom settings.

    Parameters:
    - data (pd.DataFrame): The dataframe containing documents to be chunked.
    - markdown_split (bool): Whether to split markdown headers into separate chunks.
    - hf_tokenizer_name (str, optional): Name of the Hugging Face tokenizer to use.
    - chunk_size (int, optional): Size of each chunk if not using hf_tokenizer.
    - chunk_overlap (int, optional): Overlap size between chunks if not using hf_tokenizer.
    - separators (list, optional): List of separators to use for splitting the text.

    Returns:
    - Tuple[List[Document], dict]: A tuple containing the list of processed unique document chunks and chunking information.
    """

    logging.info("Building the list of Document objects")

    # advantage of using a loader
    # No need to know which metadata are stored in the dataframe
    # Every column except page_content_column contains metadata
    document_list = DataFrameLoader(data, page_content_column="content").load()

    if kwargs.get("markdown_split", False):
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
        )
        document_list = make_md_splits(document_list, markdown_splitter)

    # Initialize token/char splitter
    text_splitter = get_text_splitter(**kwargs)

    # Split documents into chunks
    docs_processed = text_splitter.split_documents(document_list)

    # Remove duplicates
    unique_texts = set()
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts.add(doc.page_content)
            docs_processed_unique.append(doc)

    logging.info(
        f"Number of created chunks: {len(docs_processed_unique)} in the Vector Database"
    )

    return docs_processed_unique


def compute_autokenizer_chunk_size(hf_tokenizer_name: str) -> tuple:
    """
    Computes the chunk size and chunk overlap for text processing based on the
    capabilities of a Hugging Face tokenizer.

    Parameters:
    hf_tokenizer_name (str): The name of the Hugging Face tokenizer to use.

    Returns:
    tuple: A tuple containing the tokenizer instance, the chunk size, and the chunk overlap.
    """
    # Load the tokenizer
    autokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)

    # Get the maximum token length the tokenizer can handle
    chunk_size = autokenizer.model_max_length

    # Compute chunk overlap as 10% of the chunk size
    chunk_overlap = int(chunk_size * 0.1)

    return autokenizer, chunk_size, chunk_overlap


def get_text_splitter(**kwargs) -> tuple[RecursiveCharacterTextSplitter, dict]:
    """
    Get a text splitter based on the specified parameters.

    Parameters:
    use_tokenizer_to_chunk (bool): Whether to use a Hugging Face tokenizer to chunk the text.
    embedding_model (str): The name of the Hugging Face tokenizer to use.
    chunk_size (int): The size of each chunk.
    chunk_overlap (int): The overlap size between chunks.
    separators (list): List of separators to use for chunking.

    Returns:
    RecursiveCharacterTextSplitter: A text splitter instance.
    """

    if kwargs.get("use_tokenizer_to_chunk", False):
        autokenizer, chunk_size, chunk_overlap = compute_autokenizer_chunk_size(
            kwargs.get("embedding_model")
        )

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            autokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=kwargs.get("separators"),
        )
    else:
        if kwargs.get("chunk_size") is None or kwargs.get("chunk_overlap") is None:
            raise ValueError(
                "chunk_size and chunk_overlap must be specified if use_tokenizer_to_chunk is set to True"
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=kwargs.get("chunk_size"),
            chunk_overlap=kwargs.get("chunk_overlap"),
            separators=kwargs.get("separators"),
        )

    return text_splitter


def make_md_splits(
    document_list: list[Document], markdown_splitter: MarkdownHeaderTextSplitter
) -> list[Document]:
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
