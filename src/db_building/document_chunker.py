from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents.base import Document
from transformers import AutoTokenizer
import pandas as pd
import logging


def chunk_documents(
    data: pd.DataFrame, hf_tokenizer_name: str = None, chunk_size: int = None, chunk_overlap: int = None, separators: list = None
) -> tuple[list[Document], dict]:
    """
    Chunks documents from a dataframe into smaller pieces using specified tokenizer settings or custom settings.

    Parameters:
    - data (pd.DataFrame): The dataframe containing documents to be chunked.
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

    # Initialize text splitter
    text_splitter, chunk_infos = get_text_splitter(hf_tokenizer_name, chunk_size, chunk_overlap, separators)

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

    return docs_processed_unique, chunk_infos


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


def get_text_splitter(hf_tokenizer_name: str, chunk_size: int, chunk_overlap: int, separators: list) -> tuple[RecursiveCharacterTextSplitter, dict]:
    """
    Get a text splitter based on the specified parameters.

    Parameters:
    hf_tokenizer_name (str): The name of the Hugging Face tokenizer to use.
    chunk_size (int): The size of each chunk.
    chunk_overlap (int): The overlap size between chunks.
    separators (list): List of separators to use for chunking.

    Returns:
    RecursiveCharacterTextSplitter: A text splitter instance.
    """

    if hf_tokenizer_name:
        autokenizer, chunk_size, chunk_overlap = compute_autokenizer_chunk_size(hf_tokenizer_name)

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            autokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
    else:
        if chunk_size is None or chunk_overlap is None:
            raise ValueError("chunk_size and chunk_overlap must be specified if hf_tokenizer is not provided")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

    return text_splitter, {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
