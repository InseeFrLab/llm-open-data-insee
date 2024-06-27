import logging

import pandas as pd
from config import CHUNK_OVERLAP, CHUNK_SIZE, EMB_MODEL_NAME, MARKDOWN_SEPARATORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents.base import Document
from transformers import AutoTokenizer


def compute_autokonenizer_chunk_size(embedding_model: str = EMB_MODEL_NAME):
    # load a tokenizer to chunk the documents based on tokenizer abilities (get maximal token size)
    autokenizer = AutoTokenizer.from_pretrained(embedding_model)
    chunk_size = autokenizer.model_max_length
    chunk_overlap = int(chunk_size / 10)
    return autokenizer, chunk_size, chunk_overlap


def build_documents_from_dataframe(df: pd.DataFrame) -> list[Document]:
    """
    df : DataFrame containing page content
    but also additional information which are documents metadata
    """
    logging.info("Building the list of Document objects")

    loader = DataFrameLoader(df, page_content_column="content")
    # advantage of using a loader
    # No need to know which metadata are stored in the dataframe
    # Every column except page_content_column contains metadata
    document_list = loader.load()

    HF_TOKENIZER = True

    if HF_TOKENIZER:
        autokenizer, chunk_size, chunk_overlap = compute_autokonenizer_chunk_size(EMB_MODEL_NAME)

        logging.info("chunk size : ", chunk_size)
        logging.info("chunk overlap size : ", chunk_overlap)

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            autokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=MARKDOWN_SEPARATORS,
        )
    else:
        # do not take into account the  Tokenizer's specs from embeddnig model
        logging.info("chunk size : ", CHUNK_SIZE)
        logging.info("chunk overlap size : ", CHUNK_OVERLAP)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=MARKDOWN_SEPARATORS)

    docs_processed = text_splitter.split_documents(document_list)

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    logging.info(f"Number of created chuncks : {len(docs_processed_unique)} in the Vector Database")
    return docs_processed_unique
