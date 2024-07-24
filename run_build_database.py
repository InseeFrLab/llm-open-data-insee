import argparse
import ast
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PosixPath

import mlflow
import pandas as pd
import s3fs

from src.chain_building import build_chain_validator
from src.config import (
    CHROMA_DB_LOCAL_DIRECTORY,
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


# Command-line arguments
def str_to_list(arg):
    # Convert the argument string to a list
    return ast.literal_eval(arg)


# Define user-level parameters
parser = argparse.ArgumentParser(description="LLM building parameters")
parser.add_argument(
    "--experiment_name",
    type=str,
    help="""
    Name of the experiment.
    """,
    required=True,
)
parser.add_argument(
    "--data_raw_s3_path",
    type=str,
    default="data/raw_data/applishare_solr_joined.parquet",
    help="""
    Path to the raw data.
    Default to data/raw_data/applishare_solr_joined.parquet
    """,
    required=True,
)
parser.add_argument(
    "--collection_name",
    type=str,
    default="insee_data",
    help="""
    Collection name.
    Default to insee_data
    """,
    required=True,
)
parser.add_argument(
    "--markdown_split",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="""
    Should we use a markdown split ?
    --markdown_split yields True and --no-markdown_split yields False
    """,
)
parser.add_argument(
    "--use_tokenizer_to_chunk",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="""
    Should we use the tokenizer of the embedding model to chunk ?
    --use_tokenizer_to_chunk yields True and --no-use_tokenizer_to_chunk yields False
    """,
)
parser.add_argument(
    "--separators",
    type=str_to_list,
    default=r"['\n\n', '\n', '.', ' ', '']",
    help="List separators to split the text",
)
parser.add_argument(
    "--embedding_model",
    type=str,
    default="OrdalieTech/Solon-embeddings-large-0.1",
    help="""
    Embedding model.
    Should be a huggingface model.
    Defaults to OrdalieTech/Solon-embeddings-large-0.1
    """,
)
parser.add_argument(
    "--max_pages",
    type=int,
    default=None,
    help="""
    Maximum number of pages to use for the vector database.
    """,
)
parser.add_argument(
    "--llm_model",
    type=str,
    default=os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"),
    help="""
    LLM used to generate chat.
    Should be a huggingface model.
    Defaults to mistralai/Mistral-7B-Instruct-v0.2
    """,
    required=True,
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
    default=None,
    help="""
    Chunk size
    """,
)
parser.add_argument(
    "--chunk_overlap",
    type=str,
    default=None,
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


def run_build_database(
    experiment_name: str,
    data_raw_s3_path: str,
    collection_name: str,
    embedding_model: str,
    llm_model: str,
    **kwargs,
):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": f"""https://{os.environ["AWS_S3_ENDPOINT"]}"""})

        # ------------------------
        # I - BUILD VECTOR DATABASE

        db, df_raw = build_vector_database(
            data_path=data_raw_s3_path,
            persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
            embedding_model=embedding_model,
            collection_name=collection_name,
            filesystem=fs,
            **kwargs,
        )

        # Log raw dataset built from web4g
        mlflow_data_raw = mlflow.data.from_pandas(df_raw.head(10), source=f"s3://{S3_BUCKET}/{data_raw_s3_path}", name="web4g_data")
        mlflow.log_input(mlflow_data_raw, context="pre-embedding")
        mlflow.log_table(data=df_raw.head(10), artifact_file="web4g_data.json")

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
        mlflow.log_metric("number_documents", len(db_docs))

        # Log environment necessary to reproduce the experiment
        current_dir = Path(".")
        FILES_TO_LOG = (
            [PosixPath("src/build_database.py")] + list(current_dir.glob("src/db_building/*.py")) + list(current_dir.glob("src/config/*.py"))
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            for file_path in FILES_TO_LOG:
                relative_path = file_path.relative_to(current_dir)
                destination_path = tmp_dir_path / relative_path.parent
                destination_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(file_path, destination_path)

            # Generate requirements.txt using pipreqs
            subprocess.run(["pipreqs", str(tmp_dir_path)], check=True)

            # Log all Python files to MLflow artifact
            mlflow.log_artifacts(tmp_dir, artifact_path="environment")

    # ------------------------
    # II - CREATING RETRIEVER

    logging.info(f"Training retriever {80*'='}")

    # mlflow.log_param("llm_model_name", llm_model)
    # mlflow.log_param("max_new_tokens", max_new_tokens)
    # mlflow.log_param("temperature", model_temperature)
    mlflow.log_text(RAG_PROMPT_TEMPLATE, "rag_prompt.md")

    llm, tokenizer = build_llm_model(
        model_name=llm_model,
        quantization_config=kwargs.get("quantization"),
        config=True,
        token=os.getenv("HF_TOKEN"),
        streaming=False,
        generation_args={
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "return_full_text": False,
            "do_sample": False,
            "temperature": kwargs.get("model_temperature"),
        },
    )

    logging.info("Logging an example of tokenized text")
    mlflow.log_text(
        f"{query} \n ---------> \n {', '.join(tokenizer.tokenize(query))}",
        "example_tokenizer.json",
    )

    retriever, vectorstore = load_retriever(
        emb_model_name=embedding_model,
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
    true_positive_validator = validator_answers.loc[validator_answers["real"], "real"].mean()
    true_negative_validator = 1 - (validator_answers.loc[~validator_answers["real"], "real"].mean())
    mlflow.log_metric("validator_true_positive", 100 * true_positive_validator)
    mlflow.log_metric("validator_negative", 100 * true_negative_validator)

    # ------------------------
    # IV - RERANKER

    reranking_method = kwargs.get("reranking_method")

    if reranking_method is not None:
        logging.info(f"Applying reranking {80*'='}")
        logging.info(f"Selected method: {reranking_method}")
    else:
        logging.info(f"Skipping reranking since value is None {80*'='}")


if __name__ == "__main__":
    assert "MLFLOW_TRACKING_URI" in os.environ, "Please set the MLFLOW_TRACKING_URI environment variable."

    args = parser.parse_args()

    run_build_database(
        **vars(args),
    )
