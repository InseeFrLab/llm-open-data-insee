import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import s3fs

from src.chain_building import build_chain_validator
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMB_MODEL_NAME,
    RAG_PROMPT_TEMPLATE,
    S3_BUCKET,
)
from src.db_building import build_vector_database, chroma_topk_to_df, load_retriever
from src.evaluation import evaluate_question_validator
from src.model_building import build_llm_model

# Logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.DEBUG,
)


fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": f"""https://{os.environ["AWS_S3_ENDPOINT"]}"""}
)


# PARAMETERS ----------------------------------------

# Check mlflow URL is defined
assert (
    "MLFLOW_TRACKING_URI" in os.environ
), "Please set the MLFLOW_TRACKING_URI environment variable."

# Global parameters
EXPERIMENT_NAME = "BUILD_CHROMA_TEST"
MAX_NUMBER_PAGES = 20
CHROMA_DB_LOCAL_DIRECTORY = "data/chroma_database/chroma_test/"

# Define user-level parameters
parser = argparse.ArgumentParser(description="LLM building parameters")
parser.add_argument(
    "--embedding",
    type=str,
    default=EMB_MODEL_NAME,
    help="""
    Embedding model.
    Should be a huggingface model.
    Defaults to OrdalieTech/Solon-embeddings-large-0.1
    """,
)
parser.add_argument(
    "--model",
    type=str,
    default=os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"),
    help="""
    LLM used to generate chat.
    Should be a huggingface model.
    Defaults to mistralai/Mistral-7B-Instruct-v0.2
    """,
)
parser.add_argument(
    "--quantization",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="""
    Should we use a quantized version of "model" argument ?
    --quantization yields True and --no-quantization yields False
    """,
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=2000,
    help="""
    The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    See https://huggingface.co/docs/transformers/main_classes/text_generation
    """,
)
parser.add_argument(
    "--model_temperature",
    type=int,
    default=1,
    help="""
    The value used to modulate the next token probabilities.
    See https://huggingface.co/docs/transformers/main_classes/text_generation
    """,
)
parser.add_argument(
    "--chunk_size",
    type=str,
    default=CHUNK_SIZE,
    help="""
    Chunk size
    """,
)
parser.add_argument(
    "--chunk_overlap",
    type=str,
    default=CHUNK_OVERLAP,
    help="""
    Chunk overlap
    """,
)
parser.add_argument(
    "--reranking_method",
    type=str,
    default=None,
    help="""
    Reranking document relevancy after retrieval phase.
    Defaults to None (no reranking)
    """,
)


logging.info("At this time, chunk_overlap and chunk_size are ignored")

args = parser.parse_args()


# PIPELINE ----------------------------------------------------

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(EXPERIMENT_NAME)

# Build database
with mlflow.start_run() as run:

    # ------------------------
    # I - BUILD VECTOR DATABASE

    logging.info(f"Building vector database {80*'='}")

    db, data, chunk_infos = build_vector_database(
        data_path="data/raw_data/applishare_solr_joined.parquet",
        persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
        embedding_model=args.embedding,
        collection_name=COLLECTION_NAME,
        filesystem=fs,
        max_pages=MAX_NUMBER_PAGES,
    )

    # Log raw dataset built from web4g
    df_raw = pd.read_parquet(
        f"s3://{S3_BUCKET}/data/raw_data/applishare_solr_joined.parquet", filesystem=fs
    ).head(10)
    mlflow_data_raw = mlflow.data.from_pandas(
        df_raw,
        source=f"s3://{S3_BUCKET}/data/raw_data/applishare_solr_joined.parquet",
        name="web4g_data",
    )
    mlflow.log_input(mlflow_data_raw, context="pre-embedding")
    mlflow.log_table(data=df_raw, artifact_file="web4g_data.json")

    # Log the vector database
    mlflow.log_artifacts(Path(CHROMA_DB_LOCAL_DIRECTORY), artifact_path="chroma")

    # Log the first chunks of the vector database
    db_docs = db.get()["documents"]
    example_documents = pd.DataFrame(db_docs[:10], columns=["document"])
    mlflow.log_table(data=example_documents, artifact_file="example_documents.json")

    # Log a result of a similarity search
    query = "Quels sont les chiffres du chômages en 2023 ?"
    retrieved_docs = db.similarity_search(query, k=5)
    result = chroma_topk_to_df(retrieved_docs)
    mlflow.log_table(data=result, artifact_file="retrieved_documents_db_only.json")
    mlflow.log_param("question_asked", query)

    # Log parameters and metrics
    mlflow.log_param("collection_name", COLLECTION_NAME)
    mlflow.log_param("number_pages", MAX_NUMBER_PAGES)
    mlflow.log_param("embedding_model_name", args.embedding)
    mlflow.log_metric("number_documents", len(db_docs))
    for key, value in chunk_infos.items():
        mlflow.log_param(key, value)

    # Log environment necessary to reproduce the experiment
    current_dir = Path(".")
    FILES_TO_LOG = list(current_dir.glob("src/db_building/*.py")) + list(
        current_dir.glob("src/config/*.py")
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        for file_path in FILES_TO_LOG:
            relative_path = file_path.relative_to(current_dir)
            destination_path = tmp_dir_path / relative_path.parent
            destination_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, destination_path)

        requirements_path = tmp_dir_path / "requirements.txt"

        # Generate requirements.txt using pipreqs
        subprocess.run(["pipreqs", str(tmp_dir_path)], check=True)

        # Log all Python files to MLflow artifact
        mlflow.log_artifacts(tmp_dir, artifact_path="environment")

    # ------------------------
    # II - CREATING RETRIEVER

    logging.info(f"Training retriever {80*'='}")

    mlflow.log_param("llm_model_name", args.model)
    mlflow.log_param("max_new_tokens", args.max_new_tokens)
    mlflow.log_param("temperature", args.model_temperature)
    mlflow.log_text(RAG_PROMPT_TEMPLATE, "rag_prompt.md")

    llm, tokenizer = build_llm_model(
        model_name=args.model,
        quantization_config=args.quantization,
        config=True,
        token=os.getenv("HF_TOKEN"),
        streaming=False,
        generation_args={
            "max_new_tokens": args.max_new_tokens,
            "return_full_text": False,
            "do_sample": False,
            "temperature": args.model_temperature,
        },
    )

    logging.info("Logging an example of tokenized text")
    mlflow.log_text(
        f"{query} \n ---------> \n {', '.join(tokenizer.tokenize(query))}",
        "example_tokenizer.json",
    )

    retriever, vectorstore = load_retriever(
        emb_model_name=args.embedding,
        vectorstore=db,
        persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
        retriever_params={"search_type": "similarity", "search_kwargs": {"k": 30}},
    )

    # Log retriever
    retrieved_docs = retriever.invoke("Quels sont les chiffres du chômage en 2023 ?")
    result_retriever_raw = chroma_topk_to_df(retrieved_docs)
    mlflow.log_table(
        data=result_retriever_raw,
        artifact_file="retrieved_documents_retriever_raw.json",
    )

    # ------------------------
    # III - QUESTION VALIDATOR

    logging.info("Testing the questions that are accepted/refused by our agent")

    validator = build_chain_validator(evaluator_llm=llm, tokenizer=tokenizer)
    validator_answers = evaluate_question_validator(validator=validator)
    true_positive_validator = validator_answers.loc[
        validator_answers["real"], "real"
    ].mean()
    true_negative_validator = 1 - (
        validator_answers.loc[~validator_answers["real"], "real"].mean()
    )
    mlflow.log_metric("validator_true_positive", 100 * true_positive_validator)
    mlflow.log_metric("validator_negative", 100 * true_negative_validator)

    # ------------------------
    # IV - RERANKER

    reranking_method = args.reranking_method

    if reranking_method is not None:
        logging.info(f"Applying reranking {80*'='}")
        logging.info(f"Selected method: {reranking_method}")
    else:
        logging.info(f"Skipping reranking since value is None {80*'='}")
