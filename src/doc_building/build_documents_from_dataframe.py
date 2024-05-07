import logging
import pandas as pd

from typing import List
from langchain_core.documents.base import Document
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_OVERLAP, CHUNK_SIZE


def build_documents_from_dataframe(df: pd.DataFrame) -> List[Document]:
    logging.info("   Building the list of Document objects")
    loader = DataFrameLoader(df, page_content_column="titre")
    document_list = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(document_list)
