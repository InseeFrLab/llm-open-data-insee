import logging
import pandas as pd
import os 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMB_MODEL_NAME, EMB_DEVICE
from config import DB_DIR_S3

from doc_building import build_documents_from_dataframe
from .utils_db import extract_paragraphs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info(f"The database will be stored in {DB_DIR_S3}")

def build_database_from_dataframe(df: pd.DataFrame) -> Chroma:
    """
    Args:
        df (pd.DataFrame)

    Returns:
        Chroma: vector database
    """
    # rename the column names:
    not_null_filtered_df = df.rename(columns={'paragraphs': 'content',
                                        'url_source': 'source',
                                        'titles_para': 'title',
                                        'dateDiffusion': 'date_diffusion',
                                        'id_origin':'insee_id'}, errors="raise" , inplace=False)

    #chucking of documents
    all_splits = build_documents_from_dataframe(not_null_filtered_df)
    logging.info("Storing the Document objects")

    embedding_model = HuggingFaceEmbeddings(#load from sentence transformers 
            model_name=EMB_MODEL_NAME,
            model_kwargs={"device": EMB_DEVICE},
            encode_kwargs={"normalize_embeddings": True}, # set True for cosine similarity
            show_progress=False
        )

    #collection_name = "insee_data_" + str(EMB_MODEL_NAME.split("/")[-1]) 
    collection_name = "insee_data"
    db = Chroma.from_documents(
        collection_name=collection_name,
        documents=all_splits,
        persist_directory=str(DB_DIR_S3),
        embedding=embedding_model,
    )
    logging.info("The database has been built")
    db.persist()
    return db


def build_database_from_csv(path: str) -> Chroma:

    if os.path.exists(path):
        logging.info(f"The path '{path}' exists.")
        logging.info("Start building the database")

        data = pd.read_csv(path, low_memory=False)

        logging.info("Extracting paragraphs and metadata")
        df = extract_paragraphs(data)# dataframe
        
        # rename the column names:
        df.rename(columns={'paragraphs': 'content',
                            'url_source': 'source',
                            'dateDiffusion': 'date_diffusion',
                            'id_origin': 'insee_id'}, inplace=True) 

        #remove NaN value to empty strings
        logging.info("Remove NaN values by empty strings")
        df.fillna(value="", inplace=True)

        #chucking of documents
        all_splits = build_documents_from_dataframe(df)
        logging.info("Storing the Document objects")

        embedding_model = HuggingFaceEmbeddings( #load from sentence transformers 
                model_name=EMB_MODEL_NAME,
                model_kwargs={"device": EMB_DEVICE},
                encode_kwargs={"normalize_embeddings": True}, # set True for cosine similarity
                show_progress=False
            )

        #collection_name = "insee_data_" + str(EMB_MODEL_NAME.split("/")[-1]) 
        collection_name = "insee_data"
        db = Chroma.from_documents(
            collection_name=collection_name,
            documents=all_splits,
            persist_directory=str(DB_DIR_S3),
            embedding=embedding_model,
        )
        logging.info("The database has been built")
        db.persist()
        return db
    else:
        logging.info("Error Database : database File not found")
        logging.info(f"The path '{path}' does not exist.")
        return None

