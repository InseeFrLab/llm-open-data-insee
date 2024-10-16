import logging
import gc
import s3fs

from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


from src.config import (
    CHROMA_DB_LOCAL_DIRECTORY,
    COLLECTION_NAME,
    EMB_DEVICE,
    EMB_MODEL_NAME,
    S3_BUCKET,
)


from .corpus_building import (
    build_or_use_from_cache, DEFAULT_LOCATIONS,
)
from .utils_db import split_list

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_collection_name(collection_name: str):
    """
    Parse a concatenated string to extract the embedding model name, chunk size, and overlap size.
    :param concatenated_string: A string in the format 'embeddingmodelname_chunkSize_overlapSize'
    :return: A dictionary with the parsed values
    """
    try:
        # Split the string by the underscore delimiter
        parts = collection_name.split("_")

        # Ensure there are exactly three parts
        if len(parts) != 3:
            raise ValueError(
                "String format is incorrect. Expected format: 'modelname_chunkSize_overlapSize'"
            )

        # Extract and assign the parts
        model_name = parts[0]
        chunk_size = int(parts[1])
        overlap_size = int(parts[2])

        # Return the parsed values in a dictionary
        return {
            "model_name": model_name,
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
        }
    except Exception as e:
        print(f"Error parsing string: {e}")
        return None


# BUILD VECTOR DATABASE FROM COLLECTION -------------------------


def build_vector_database(
    model_id: str,
    persist_directory: str,
    collection_name: str,
    filesystem: s3fs.S3FileSystem,
    s3_bucket: str = S3_BUCKET,
    location_dataset: dict = DEFAULT_LOCATIONS,
    **kwargs,
) -> Chroma:

    logging.info(f"The database will temporarily be stored in {persist_directory}")

    logging.info("Start building the database")

    # Call the process_data function to handle data loading, parsing, and splitting
    df, all_splits = build_or_use_from_cache(
        filesystem=filesystem,
        s3_bucket=s3_bucket,
        location_dataset=location_dataset,
        model_id=model_id,
        **kwargs
    )

    logging.info("Document chunking is over, starting to embed them")

    # Building embedding model using parameters from kwargs
    embedding_model = kwargs.get("embedding_model")
    embedding_device = kwargs.get("embedding_device")

    logging.info("Loading embedding model")

    emb_model = HuggingFaceEmbeddings(  # load from sentence transformers
        model_name=embedding_model,
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        show_progress=False,
    )

    logging.info(f"Building embedding model: {embedding_model} on {embedding_device}")

    max_batch_size = 41600
    split_docs_chunked = split_list(all_splits, max_batch_size)

    # Loop through the chunks and build the Chroma database
    try:
        for split_docs_chunk in split_docs_chunked:
            db = Chroma.from_documents(
                collection_name=collection_name,
                documents=split_docs_chunk,
                persist_directory=persist_directory,
                embedding=emb_model,
                client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
            )
    except Exception as e:
        logging.error(f"An error occurred while building the Chroma database: {e}")

        # Return None along with the dataframe in case of failure
        return None, df

    # Cleanup after successful execution
    del emb_model
    gc.collect()

    logging.info("The database has been built")
    return db, df


# RELOAD VECTOR DATABASE FROM DIRECTORY -------------------------


def reload_database_from_local_dir(
    embed_model_name: str = EMB_MODEL_NAME,
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = CHROMA_DB_LOCAL_DIRECTORY,
    embed_device: str = EMB_DEVICE,
) -> Chroma:
    emb_model = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        multi_process=False,
        model_kwargs={"device": embed_device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )
    db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=emb_model,
    )

    logging.info(
        f"The database (collection {collection_name}) "
        f"has been reloaded from directory {persist_directory}"
    )
    return db


# LOAD RETRIEVER -------------------------------

def load_retriever(
    emb_model_name,
    vectorstore=None,
    persist_directory="data/chroma_db",
    device="cuda",
    collection_name: str = "insee_data",
    retriever_params: dict = None,
):
    # Load vector database
    if vectorstore is None:
        logging.info("Reloading database in session")
        vectorstore = reload_database_from_local_dir(
            embed_model_name=emb_model_name,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embed_device=device,
        )
    else:
        logging.info("vectorstore being provided, skipping the reloading")

    if retriever_params is None:
        retriever_params = {"search_type": "similarity", "search_kwargs": {"k": 30}}

    search_kwargs = retriever_params.get("search_kwargs", {"k": 20})

    # Set up a retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs=search_kwargs
    )
    return retriever, vectorstore
