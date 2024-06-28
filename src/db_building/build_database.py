import logging
import os

import pandas as pd
import s3fs
from chromadb.config import Settings
from config import COLLECTION_NAME, DB_DIR_LOCAL, DB_DIR_S3, EMB_DEVICE, EMB_MODEL_NAME, S3_BUCKET
from doc_building import build_documents_from_dataframe
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from .utils_db import extract_paragraphs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# TODO : pourquoi on a les config dans ce fichier ?


def build_database_from_dataframe(
    df: pd.DataFrame,
    persist_directory: str = str(DB_DIR_S3),
) -> Chroma:
    """
    Args:
        df (pd.DataFrame)

    Returns:
        Chroma: vector database
    """
    logging.info(f"The database will be stored in {persist_directory}")
    # rename the column names:
    not_null_filtered_df = df.rename(
        columns={
            "paragraphs": "content",
            "url_source": "source",
            "titles_para": "title",
            "dateDiffusion": "date_diffusion",
            "id_origin": "insee_id",
        },
        errors="raise",
        inplace=False,
    )

    # chucking of documents
    all_splits = build_documents_from_dataframe(not_null_filtered_df)
    logging.info("Storing the Document objects")

    embedding_model = HuggingFaceEmbeddings(  # load from sentence transformers
        model_name=EMB_MODEL_NAME,
        model_kwargs={"device": EMB_DEVICE},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        show_progress=False,
    )

    # collection_name = "insee_data_" + str(EMB_MODEL_NAME.split("/")[-1])
    collection_name = COLLECTION_NAME
    db = Chroma.from_documents(
        collection_name=collection_name,
        documents=all_splits,
        persist_directory=persist_directory,
        embedding=embedding_model,
    )
    logging.info("The database has been built")
    db.persist()
    return db


def build_vector_database(
    data_path: str,
    persist_directory: str,
    embedding_model: str,
    collection_name: str,
    filesystem: s3fs.S3FileSystem,
    max_pages: str = None,
) -> Chroma:
    logging.info(f"The database will temporarily be stored in {persist_directory}")

    if os.path.exists(data_path):
        logging.info("Start building the database")

        data = pd.read_parquet(f"s3://{S3_BUCKET}/{data_path}", filesystem=filesystem)

        if max_pages is not None:
            data = data.head(max_pages)

        logging.info("Extracting paragraphs and metadata")
        df = extract_paragraphs(data)

        # rename the column names:
        df.rename(
            columns={
                "paragraphs": "content",
                "url_source": "source",
                "dateDiffusion": "date_diffusion",
                "id_origin": "insee_id",
            },
            inplace=True,
        )

        # remove NaN value to empty strings
        logging.info("Remove NaN values by empty strings")
        df.fillna(value="", inplace=True)

        # chucking of documents
        all_splits = build_documents_from_dataframe(df)
        logging.info("Storing the Document objects")

        embedding_model = HuggingFaceEmbeddings(  # load from sentence transformers
            model_name=embedding_model,
            model_kwargs={"device": EMB_DEVICE},
            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
            show_progress=False,
        )

        db = Chroma.from_documents(
            collection_name=collection_name,
            documents=all_splits,
            persist_directory=persist_directory,
            embedding=embedding_model,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        logging.info("The database has been built")
        db.persist()
        return db
    else:
        logging.info("Error Database : database File not found")
        logging.info(f"The path '{data_path}' does not exist.")
        return None


def reload_database_from_local_dir(
    embed_model_name: str = EMB_MODEL_NAME,
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = DB_DIR_LOCAL,
    embed_device: str = EMB_DEVICE,
) -> Chroma:
    embedding_model = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        multi_process=True,
        model_kwargs={"device": embed_device},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )
    db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )

    logging.info(f"The database (collection {collection_name}) " f"has been reloaded from directory {persist_directory}")
    return db
