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
from chain_building import build_chain_validator
from config import COLLECTION_NAME, EMB_MODEL_NAME, RAG_PROMPT_TEMPLATE, S3_BUCKET
from db_building import build_vector_database, load_retriever
from model_building import build_llm_model

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
    default=None,
    help="""
    Embedding model.
    Should be a huggingface model.
    Defaults to OrdalieTech/Solon-embeddings-large-0.1
    """,
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
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

args = parser.parse_args()

# Replacing None with default values
if args.embedding is None:
    args.embedding = EMB_MODEL_NAME

if args.model is None:
    args.model = os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")


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

    result_list = []
    for doc in retrieved_docs:
        row = {"page_content": doc.page_content}
        row.update(doc.metadata)
        result_list.append(row)
    result = pd.DataFrame(result_list)
    mlflow.log_table(data=result, artifact_file="retrieved_documents.json")
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
    example_text = "Quels sont les chiffres du chômage ?"
    mlflow.log_text(
        f"{example_text} \n ---------> \n {', '.join(tokenizer.tokenize(example_text))}",
        "example_tokenizer.json",
    )

    retriever = load_retriever(
        emb_model_name=args.embedding,
        vectorstore=db,
        persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
        retriever_params={"search_type": "similarity", "search_kwargs": {"k": 30}},
    )

    validator = build_chain_validator(evaluator_llm=llm, tokenizer=tokenizer)
