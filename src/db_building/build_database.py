import logging
import os
import pandas as pd

from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMB_MODEL_NAME, EMB_DEVICE, COLLECTION_NAME, DB_DIR_S3, DB_DIR_LOCAL

from evaluation import RetrievalConfiguration

from doc_building import (
    build_documents_from_dataframe,
    compute_autokonenizer_chunk_size,
)
from .utils_db import extract_paragraphs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

alias_chunk_size = compute_autokonenizer_chunk_size

def build_database_from_dataframe(
    df: pd.DataFrame,
    persist_directory: str = str(DB_DIR_S3),
    embedding_model_name: str = str(EMB_MODEL_NAME),
    collection_name: str = COLLECTION_NAME,
    max_pages: str = None,
    config: RetrievalConfiguration = None
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
        errors="ignore",
        inplace=False,
    )

    # chucking of documents
    all_splits = build_documents_from_dataframe(
        not_null_filtered_df, embedding_model_name=embedding_model_name, config=config
    )
    logging.info("Storing the Document objects")

    embedding_model = HuggingFaceEmbeddings(  # load from sentence transformers
        model_name=embedding_model_name,
        model_kwargs={"device": EMB_DEVICE, "trust_remote_code" : True},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        show_progress=False
    )

    db = Chroma.from_documents(
        collection_name=collection_name,
        documents=all_splits,
        persist_directory=persist_directory,
        embedding=embedding_model,
    )
    logging.info("The database has been built")
    return db


def build_database_from_csv(
    path: str,
    persist_directory: str = str(DB_DIR_S3),
    embedding_model: str = str(EMB_MODEL_NAME),
    collection_name: str = COLLECTION_NAME,
    max_pages: str = None,
    config: RetrievalConfiguration = None

) -> Chroma:
    logging.info(f"The database will be stored in {persist_directory}")

    if os.path.exists(path):
        logging.info(f"The path '{path}' exists.")
        logging.info("Start building the database")

        data = pd.read_csv(path, low_memory=False)
        if max_pages is not None:
            data = data.head(max_pages)

        logging.info("Extracting paragraphs and metadata")
        df = extract_paragraphs(data)  # dataframe

        # rename the column names:
        df.rename(
            columns={
                "paragraphs": "content",
                "url_source": "source",
                "dateDiffusion": "date_diffusion",
                "id_origin": "insee_id",
            },
            inplace=True,
            errors="ignore",
        )

        # remove NaN value to empty strings
        logging.info("Remove NaN values by empty strings")
        df.fillna(value="", inplace=True)

        # chucking of documents
        all_splits = build_documents_from_dataframe(
            df, embedding_model_name=embedding_model, config=config
        )
        logging.info("Storing the Document objects")

        embedding_model = HuggingFaceEmbeddings(  # load from sentence transformers
            model_name=embedding_model,
            multi_process=False,
            model_kwargs={"device": EMB_DEVICE, "trust_remote_code": True},
            encode_kwargs={
                "normalize_embeddings": True
            },  # set True for cosine similarity
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
        return db
    else:
        logging.info("Error Database : database File not found")
        logging.info(f"The path '{path}' does not exist.")
        return None


def reload_database_from_local_dir(
    embed_model_name: str = EMB_MODEL_NAME,
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = DB_DIR_LOCAL,
    embed_device: str = EMB_DEVICE,
    config: RetrievalConfiguration = None
) -> Chroma:

    if config is not None:
        info = parse_collection_name(collection_name)
        if info is not None:
            config.chunk_size = info.get("chunk_size")
            config.overlap_size = info.get("overlap_size")

    embedding_model = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        multi_process=False,
        model_kwargs={"device": embed_device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )
    db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )

    logging.info(
        f"The database (collection {collection_name}) "
        f"has been reloaded from directory {persist_directory}"
    )
    return db

def parse_collection_name(collection_name: str):
    """
    Parse a concatenated string to extract the embedding model name, chunk size, and overlap size.

    :param concatenated_string: A string in the format 'embeddingmodelname_chunkSize_overlapSize'
    :return: A dictionary with the parsed values
    """
    try:
        # Split the string by the underscore delimiter
        parts = collection_name.split('_')
        
        # Ensure there are exactly three parts
        if len(parts) != 3:
            raise ValueError("String format is incorrect. Expected format: 'modelname_chunkSize_overlapSize'")
        
        # Extract and assign the parts
        model_name = parts[0]
        chunk_size = int(parts[1])
        overlap_size = int(parts[2])
        
        # Return the parsed values in a dictionary
        return {
            "model_name": model_name,
            "chunk_size": chunk_size,
            "overlap_size": overlap_size
        }
    except Exception as e:
        print(f"Error parsing string: {e}")
        return None
