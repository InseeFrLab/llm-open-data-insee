import argparse

import mlflow
import pandas as pd
import s3fs
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from src.config import set_config
from src.data.caching import chunk_documents_or_load_from_cache, parse_documents_or_load_from_cache
from src.evaluation.basic_evaluation import answer_faq_by_bot, transform_answers_bot
from src.model.prompt import similarity_search_instructions
from src.results_logging.mlflow_utils import mlflow_log_source_files
from src.utils.utils_vllm import get_models_from_env
from src.vectordatabase.chroma import chroma_vectorstore_as_retriever
from src.vectordatabase.client import create_client_and_collection, get_number_docs_collection
from src.vectordatabase.embed_by_piece import chunk_documents_and_store
from src.vectordatabase.output_parsing import langchain_documents_to_df
from src.vectordatabase.qdrant import (
    create_collection_alias_qrant,
    create_snapshot_collection_qdrant,
    qdrant_vectorstore_as_retriever,
)

load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Parameters to control database building")
parser.add_argument(
    "--collection_name",
    type=str,
    default="dirag_experimentation",
    help="Database collection name (for Qdrant or Chroma like database)",
)
parser.add_argument(
    "--max_pages",
    type=int,
    default=None,
    help="Max number of pages that should be considered",
)
parser.add_argument(
    "--mlflow_experiment_name",
    type=str,
    default="vector_database_building",
    help="Experiment name in mlflow",
)
parser.add_argument(
    "--chunking_strategy",
    type=str,
    default="None",
    choices=["None", "Recursive", "recursive", "Character", "character"],
    help="Chunking strategy for documents"
    "If None (default), corpus is left asis. "
    "If recursive, use Langchain's CharacterTextSplitter",
)
parser.add_argument(
    "--max_document_size",
    type=int,
    default=1200,
    help="Threshold size for documents. "
    "If None (default), corpus is left asis. "
    "If value provided, passed to Langchain's CharacterTextSplitter is applied",
)
parser.add_argument(
    "--chunk_overlap",
    type=int,
    default=100,
    help="Chunk overlap when documents are split. "
    "If value provided, passed to Langchain's CharacterTextSplitter is applied",
)
parser.add_argument(
    "--dataset",
    choices=["dirag", "complete"],
    default="complete",
    help="""
        Choose the dataset type: 'dirag' for restricted DIRAG data,
        'complete' for the full web4g dataset (default: 'complete'),
    """,
)
parser.add_argument(
    "--log_database_snapshot", action="store_true", help="Should we log database snapshot ? (default: False)"
)
parser.add_argument(
    "--top_k_statistics",
    type=int,
    default=10,
    help="Number of documents that should be given by retriever for evaluation",
)
parser.add_argument(
    "--database_engine",
    type=str,
    default="qdrant",
    help="Vector database engine",
)
parser.add_argument("--verbose", action="store_true", help="Enable verbose output (default: False)")


# Example usage:
# python run_build_dataset.py max_pages 10 --dataset dirag
# python run_build_dataset.py max_pages 10

args = parser.parse_args()

# Logging configuration
# logger = logging.getLogger(__name__)

# CONFIGURATION ------------------------------------------

engine = args.database_engine

config = set_config(
    use_vault=True,
    components=["s3", "mlflow", "database", "model"],
    mlflow_experiment_name=args.mlflow_experiment_name,
    database_manager=engine,
    models_location={
        "url_embedding_model": "ENV_URL_EMBEDDING_MODEL",
        "url_generative_model": "ENV_URL_GENERATIVE_MODEL",
    },
    override={f"{engine}_COLLECTION_NAME": args.collection_name},
    verbose=args.verbose,
)


filtered_config = {k: v for k, v in config.items() if not any(s in k.lower() for s in ["key", "token", "secret"])}

embedding_model = get_models_from_env(url_embedding="URL_EMBEDDING_MODEL", config_dict=config).get("embedding")

url_database_client = config.get(f"{engine.upper()}_URL")
api_key_database_client = config.get(f"{engine.upper()}_API_KEY")


logger.debug(f"Using {embedding_model} for database retrieval")
logger.debug(f"Setting {url_database_client} as vector database endpoint")


# PARAMETERS ------------------------------------------

S3_PATH = "s3://projet-llm-insee-open-data/data/raw_data/applishare_solr_joined.parquet"
collection_name = args.collection_name
s3_bucket = "projet-llm-insee-open-data"
faq_s3_path = "data/FAQ_site/faq.parquet"

max_document_size = args.max_document_size
chunk_overlap = args.chunk_overlap

filesystem = s3fs.S3FileSystem(endpoint_url=config.get("endpoint_url"))


corpus_constructor_args = {
    "field": args.dataset,
    "web4g_path_uri": S3_PATH,
    "fs": filesystem,
    "search_cols": ["titre", "libelleAffichageGeo", "xml_intertitre"],
}

chunking_args = {"strategy": args.chunking_strategy, "chunk_size": max_document_size, "chunk_overlap": chunk_overlap}

parameters_database_construction = {
    "embedding_model": embedding_model,
    f"{engine.upper()}_URL_API": url_database_client,
    "chunking_strategy": args.chunking_strategy,
    "max_document_size": max_document_size,
    "chunk_overlap": chunk_overlap,
    "engine": engine,
}


CACHE_DIR = f"s3://projet-llm-insee-open-data/data/intermediate/dataset={args.dataset}"
path_cached_parsed_documents = f"{CACHE_DIR}/web4g_parsed/parsed_data.parquet"
path_cached_chunked_documents = (
    f"{CACHE_DIR}/web4g_chunked/"
    f"strategy={args.chunking_strategy}/chunk_size={max_document_size}/"
    f"chunk_overlap={chunk_overlap}/data.jsonl"
)

if args.dataset == "dirag":
    logger.warning("Restricting publications to DIRAG related content")


print(chunking_args)

# MAIN PIPELINE --------------------------------------


def run_build_database() -> None:
    mlflow.set_tracking_uri(config.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(config.get("MLFLOW_EXPERIMENT_NAME"))

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                **filtered_config,
                **parameters_database_construction,
                **{"dataset": args.dataset, "max_pages": args.max_pages, "max_document_size": max_document_size},
            }
        )

        # PARSING CORPUS -----------------------------------

        logger.debug(f"Importing and parsing raw data {30 * '-'}")

        data = parse_documents_or_load_from_cache(
            path_for_cache=path_cached_parsed_documents,
            load_from_cache=True,
            max_pages=args.max_pages,
            filesystem=filesystem,
            corpus_constructor_args=corpus_constructor_args,
        )

        data["mlflow_run_id"] = run_id  # Add mlflow run id in metadata

        # CHUNKING DOCUMENTS --------------------------

        loader = DataFrameLoader(data, page_content_column="content")
        documents = loader.load()

        logger.info(f"Before chunking, we have {len(documents)} pages")

        documents = chunk_documents_or_load_from_cache(
            documents_before_chunking=documents,
            path_for_cache=path_cached_chunked_documents,
            max_pages=args.max_pages,
            load_from_cache=True,
            filesystem=filesystem,
            chunking_args=chunking_args,
        )

        logger.info(f"After chunking, we have {len(documents)} documents")

        # CREATE DATABASE COLLECTION -----------------------

        logger.info("Connecting to vector database")

        emb_model = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=config.get("OPENAI_API_BASE_EMBEDDING"),
            openai_api_key=config.get("OPENAI_API_KEY_EMBEDDING"),
            tiktoken_enabled=False,
        )

        model_max_len = len(emb_model.embed_query("retrieving hidden_size"))
        # confusion between hidden_size and model_max_len
        # get_model_max_len(model_id=embedding_model)

        unique_collection_name = f"{collection_name}_{run_id}"

        client = create_client_and_collection(
            url=url_database_client,
            api_key=api_key_database_client,
            collection_name=unique_collection_name,
            model_max_len=model_max_len,
            engine=engine,
            vector_name=embedding_model,
        )

        # EMBEDDING DOCUMENTS IN VECTOR DATABASE -----------------------
        logger.info("Putting documents in vector database")

        chunk_documents_and_store(
            documents,
            emb_model,
            collection_name=unique_collection_name,
            url=url_database_client,
            api_key=api_key_database_client,
            engine=engine,
            client=client,
        )

        # SETTING ALIAS -----------------------

        if engine == "qdrant":
            create_collection_alias_qrant(
                client=client, initial_collection_name=unique_collection_name, alias_collection_name=collection_name
            )

        mlflow.log_params(
            {
                "COLLECTION_UNIQUE": unique_collection_name,
                "COLLECTION_ALIAS": collection_name,
            }
        )

        # LOGGING DATABASE STATISTICS --------------------------

        n_documents = get_number_docs_collection(client=client, collection_name=unique_collection_name, engine=engine)

        dict_metadata_collection = {"n_documents": n_documents}

        if engine == "qdrant":
            collection_info = client.get_collection(collection_name=unique_collection_name)
            embedding_size = collection_info.config.params.vectors.get(embedding_model).size
            dict_metadata_collection = {
                **{"embedding_size": embedding_size, "embedding_model": embedding_model},
                **dict_metadata_collection,
            }

        mlflow.log_params(dict_metadata_collection)

        # CREATING SNAPSHOT FOR LOGGING -------------------

        if args.log_database_snapshot is True and engine == "qdrant":
            logger.info("Logging database snapshot")

            create_snapshot_collection_qdrant(
                client=client,
                collection_name=unique_collection_name,
                url=url_database_client,
                api_key=api_key_database_client,
            )

            logger.success("Database building successful")

        # TURNING DATABASE INTO RETRIEVER ----------------------

        constructor_retriever = qdrant_vectorstore_as_retriever
        if engine == "chroma":
            constructor_retriever = chroma_vectorstore_as_retriever

        retriever = constructor_retriever(
            client=client,
            collection_name=unique_collection_name,
            embedding_function=emb_model,
            vector_name=emb_model.model,
            number_retrieved_docs=10,
        )

        # Log a result of a similarity search
        test_query = "Quels sont les chiffres du chômage en 2023 ?"
        query = f"{similarity_search_instructions}\nQuery: {test_query}"
        mlflow.log_param("prompt_retriever", similarity_search_instructions)
        retrieved_docs = retriever.invoke(query)
        result_retriever_raw = langchain_documents_to_df(retrieved_docs)

        mlflow.log_table(data=result_retriever_raw, artifact_file="retrieved_documents.json")
        mlflow.log_param("question_asked", query)

        # PART II : EVALUATION ---------------------------------

        logger.info("Importing evaluation dataset")

        faq = pd.read_parquet(f"s3://{s3_bucket}/{faq_s3_path}", filesystem=filesystem)
        # Extract all URLs from the 'sources' column
        faq["urls"] = faq["sources"].str.findall(r"https?://www\.insee\.fr[^\s]*").apply(lambda s: ", ".join(s))

        # --------------------------------
        # III - RETRIEVER STATISTICS

        logger.info("Evaluating model performance against expectations")

        answers_bot = answer_faq_by_bot(retriever, faq)
        eval_reponses_bot, answers_bot_topk = transform_answers_bot(answers_bot, k=args.top_k_statistics)

        document_among_topk = answers_bot_topk["cumsum_url_expected"].max()
        document_is_top = answers_bot_topk["cumsum_url_expected"].min()

        mlflow.log_metric("document_is_first", 100 * document_is_top)
        mlflow.log_metric("document_among_topk", 100 * document_among_topk)
        mlflow.log_metrics(
            {
                f"document_in_top_{int(row['document_position'])}": 100 * row["cumsum_url_expected"]
                for _, row in answers_bot_topk.iterrows()
            }
        )
        mlflow.log_table(data=eval_reponses_bot, artifact_file="output/eval_reponses_bot.json")

        # LOGGING OTHER USEFUL THINGS --------------------------

        logger.info("Storing additional metadata")

        # Store FAQ
        mlflow_faq_raw = mlflow.data.from_pandas(faq, source=faq_s3_path, name="FAQ_data")
        mlflow.log_input(mlflow_faq_raw, context="faq-raw")
        mlflow.log_table(data=faq, artifact_file="faq_data.json")

        # Log raw dataset built from web4g
        mlflow_data_raw = mlflow.data.from_pandas(
            data.head(10),
            source=S3_PATH,
            name="web4g_data",
        )
        mlflow.log_input(mlflow_data_raw, context="pre-embedding")
        mlflow.log_table(data=data.head(10), artifact_file="web4g_data.json")

        # Log src/*.py and run_build_database.py
        mlflow_log_source_files()

        logger.info("Program ended with success.")


if __name__ == "__main__":
    run_build_database()
