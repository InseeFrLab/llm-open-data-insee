import os
from datetime import datetime

import pandas as pd
import s3fs
import streamlit as st
import torch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from src.app.utils import generate_answer_from_context, initialize_clients
from src.config import set_config
from src.model.prompt import question_instructions
from src.utils.utils_vllm import get_models_from_env

from src.vectordatabase.output_parsing import format_docs, langchain_documents_to_df
from loguru import logger

from src.model.prompt import question_instructions
from shiny.express import input

# ---------------- CONFIGURATION ---------------- #

ENGINE = "qdrant"
USE_RERANKING = True

config = set_config(
    use_vault=True,
    components=["s3", "mlflow", "database", "model"],
    models_location={
        "url_embedding_model": "ENV_URL_EMBEDDING_MODEL",
        "url_generative_model": "ENV_URL_GENERATIVE_MODEL",
        "url_reranking_model": "ENV_URL_RERANKING_MODEL",
    },
    database_manager=ENGINE,
    # override={"QDRANT_COLLECTION_NAME": "dirag_experimentation_d9867c0409cf44e1b222f9f5ede05c06"},
)

fs = s3fs.S3FileSystem(endpoint_url=config.get("endpoint_url"))
path_log = os.getenv("PATH_LOG_APP")

models = get_models_from_env(
    url_embedding="URL_EMBEDDING_MODEL", url_generative="URL_GENERATIVE_MODEL", url_reranking="URL_RERANKING_MODEL"
)
embedding_model = models.get("embedding")
generative_model = models.get("completion")
reranking_model = models.get("reranking")

retriever, _ = initialize_clients(
    config=config,
    embedding_model=embedding_model,
    use_reranking=False,
    url_reranker=os.getenv("URL_RERANKING_MODEL"),
    model_reranker=models.get("reranking"),
    engine=ENGINE
)

# ------------------------------------------------------------------------------------
# A basic Shiny Chat example powered by OpenAI.
# ------------------------------------------------------------------------------------
import os

from dotenv import load_dotenv
from chatlas import ChatOpenAI

from shiny.express import ui

with open("./prompt/system.md", "r", encoding="utf-8") as f:
    system_prompt = f.read()

with open("./prompt/question.md", "r", encoding="utf-8") as f:
    question_prompt = f.read()

prompt = PromptTemplate.from_template(question_prompt)


chat_client = ChatOpenAI(
    base_url=config.get("OPENAI_API_BASE_GENERATIVE"),
    api_key="EMPTY",
    model=generative_model,
    system_prompt=system_prompt,
)

logger.debug(system_prompt)


# Set some Shiny page options
ui.page_opts(
    title="Insee assistant Chat",
    fillable=True,
    fillable_mobile=True,
)

ui.input_switch("switch", "RAG", True)

# Create and display a Shiny chat component
chat = ui.Chat(
    id="chat",
    messages=["Posez moi une question sur les publications de l'Insee"],
)
chat.ui()


# Generate a response when the user submits a message
@chat.on_user_submit
async def handle_user_input(user_input: str):
    if input.switch() is True:
        best_documents = retriever.invoke(user_input)
        best_documents_df = langchain_documents_to_df(best_documents)
        logger.debug(user_input)
        logger.debug(best_documents_df)
        context = format_docs(best_documents)
        question_with_context = prompt.format(question=user_input, context=context)

    else:
        question_with_context = user_input
    logger.debug(question_with_context)
    response = await chat_client.stream_async(question_with_context)
    await chat.append_message_stream(response)
