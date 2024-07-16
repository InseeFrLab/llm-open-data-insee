import logging

import pandas as pd
import s3fs
from chromadb.config import Settings
from config import COLLECTION_NAME, DB_DIR_LOCAL, EMB_DEVICE, EMB_MODEL_NAME, MARKDOWN_SEPARATORS, S3_BUCKET
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from .document_chunker import chunk_documents
from .utils_db import parse_xmls, split_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_vector_database(
    data_path: str,
    persist_directory: str,
    embedding_model: str,
    collection_name: str,
    filesystem: s3fs.S3FileSystem,
    max_pages: str = None,
) -> Chroma:
    logging.info(f"The database will temporarily be stored in {persist_directory}")
    logging.info("Start building the database")

    data = pd.read_parquet(f"s3://{S3_BUCKET}/{data_path}", filesystem=filesystem)

    if max_pages is not None:
        data = data.head(max_pages)

    # Parse the XML content
    parsed_pages = parse_xmls(data)

    df = data.set_index("id").merge(pd.DataFrame(parsed_pages), left_index=True, right_index=True)
    df = df[["titre", "categorie", "url", "dateDiffusion", "theme", "collection", "libelleAffichageGeo", "content"]]

    # Temporary solution to add the RMES data
    data_path_rmes = "data/processed_data/rmes_sources_content.parquet"
    data_rmes = pd.read_parquet(f"s3://{S3_BUCKET}/{data_path_rmes}", filesystem=filesystem)
    df = pd.concat([df, data_rmes])

    # fill NaN values with empty strings since metadata doesn't accept NoneType in Chroma
    df.fillna(value="", inplace=True)

    # chucking of documents
    all_splits, chunk_infos = chunk_documents(data=df, md_split=True, hf_tokenizer_name=embedding_model, separators=MARKDOWN_SEPARATORS)

    embedding_model = HuggingFaceEmbeddings(  # load from sentence transformers
        model_name=embedding_model,
        model_kwargs={"device": EMB_DEVICE},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        show_progress=False,
    )

    split_docs_chunked = split_list(all_splits, 41000)  # Max batch size is 41666

    for split_docs_chunk in split_docs_chunked:
        db = Chroma.from_documents(
            collection_name=collection_name,
            documents=split_docs_chunk,
            persist_directory=persist_directory,
            embedding=embedding_model,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )

    logging.info("The database has been built")
    return db, data, chunk_infos


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
