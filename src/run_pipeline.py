import os
import mlflow
from model_building import build_llm_model
from chain_building import load_retriever, build_chain
from utils import loading_utilities
from langchain_core.prompts import PromptTemplate

from config import DB_DIR, DB_DIR_LOCAL, MODEL_NAME, EMB_MODEL_NAME


# PROMPT Template
RAG_PROMPT_TEMPLATE = """
<s>[INST]
Tu es un assistant spécialisé dans la statistique publique répondant aux questions d'agent de l'INSEE.
Réponds en Français seulement.
Utilise les informations obtenues dans le contexte, réponds de manière argumentée à la question posée.
La réponse doit être développée et citer ses sources.

Si tu ne peux pas induire ta réponse du contexte, ne réponds pas.
Voici le contexte sur lequel tu dois baser ta réponse :
Contexte: {context}
        ---
Voici la question à laquelle tu dois répondre :
Question: {question}
[/INST]
"""


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
    loading_utilities.load_chroma_db(s3_path=DB_DIR, persist_directory=DB_DIR_LOCAL)

    # Generate prompt template
    prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)

    # Create a pipeline with tokenizer and LLM
    llm = build_llm_model(quantization_config=True, config=True, token=os.environ["HF_TOKEN"])

    retriever = load_retriever(DB_DIR_LOCAL)

    llm_chain = build_chain(retriever, prompt, llm)

    question = "Je cherche à connaitre le bombre (et eventuellement les caractéristiques) des véhicules 'primes à la conversion' dans plusieurs départements d'occitanie, en particulier l'aveyron."
    llm_chain.invoke(question)

    mlflow.log_param("DB_DIR", DB_DIR)
    mlflow.log_param("RAG_PROMPT_TEMPLATE", RAG_PROMPT_TEMPLATE)
    mlflow.log_param("LLM_NAME", MODEL_NAME)
    mlflow.log_param("EMB_MODEL_NAME", EMB_MODEL_NAME)
    # TODO: Rajouter ce qu'il y a dans les logs + réponse
