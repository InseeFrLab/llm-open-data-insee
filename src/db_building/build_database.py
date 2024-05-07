import logging
import pandas as pd

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMB_MODEL_NAME, EMB_DEVICE
from config import DB_DIR

from doc_building import build_documents_from_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info(f"The database will be stored in {DB_DIR}")


def build_database_from_dataframe(df: pd.DataFrame):
    logging.info("Start building the database")
    not_null_df = df[(df["titre"].notnull()) & (df["id"].notnull())]
    not_null_filtered_df = not_null_df.loc[:, ["titre", "id"]]
    all_splits = build_documents_from_dataframe(not_null_filtered_df)
    logging.info("   Storing the Document objects")
    db = Chroma.from_documents(
        documents=all_splits,
        persist_directory=str(DB_DIR),
        embedding=HuggingFaceEmbeddings(
            model_name=EMB_MODEL_NAME, model_kwargs={"device": EMB_DEVICE}
        ),
    )
    logging.info("The database has been built")
    db.persist()
    return db
