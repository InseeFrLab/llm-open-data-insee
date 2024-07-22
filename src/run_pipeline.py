# CE SCRIPT N'EST PAS A JOUR !!!
# ------------------------------

import os

import mlflow
from chain_building import build_chain, load_retriever
from config import (
    DB_DIR_LOCAL,
    DB_DIR_S3,
    EMB_MODEL_NAME,
    MODEL_NAME,
    RAG_PROMPT_TEMPLATE,
)
from langchain_core.prompts import PromptTemplate
from model_building import build_llm_model
from utils import loading_utilities

EXPERIMENT_NAME = "CHAIN"

assert (
    "MLFLOW_TRACKING_URI" in os.environ
), "Please set the MLFLOW_TRACKING_URI environment variable."
assert (
    "MLFLOW_S3_ENDPOINT_URL" in os.environ
), "Please set the MLFLOW_S3_ENDPOINT_URL environment variable."
assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable."


mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run() as run:
    # Load Chroma DB
    loading_utilities.load_chroma_db(s3_path=DB_DIR_S3, persist_directory=DB_DIR_LOCAL)

    # Generate prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE
    )

    # Create a pipeline with tokenizer and LLM
    llm = build_llm_model(
        quantization_config=True, config=True, token=os.environ["HF_TOKEN"]
    )

    retriever = load_retriever(DB_DIR_LOCAL)

    llm_chain = build_chain(retriever, prompt, llm)

    question = "Je cherche à connaitre le nombre (et eventuellement les caractéristiques) des véhicules \
                'primes à la conversion' dans plusieurs départements d'occitanie, en particulier l'aveyron."
    llm_chain.invoke(question)

    mlflow.log_param("DB_DIR_S3", DB_DIR_S3)
    mlflow.log_param("RAG_PROMPT_TEMPLATE", RAG_PROMPT_TEMPLATE)
    mlflow.log_param("LLM_NAME", MODEL_NAME)
    mlflow.log_param("EMB_MODEL_NAME", EMB_MODEL_NAME)
    # TODO: Rajouter ce qu'il y a dans les logs + réponse
