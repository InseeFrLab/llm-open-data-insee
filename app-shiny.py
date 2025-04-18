
import os
from datetime import datetime

import pandas as pd
import s3fs
import streamlit as st
import torch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from src.app.feedbacks import feedback_titles, render_feedback_section
from src.app.history import activate_old_conversation, create_unique_id, summarize_conversation
from src.app.utils import generate_answer_from_context, initialize_clients
from src.config import set_config
from src.model.prompt import question_instructions
from src.utils.utils_vllm import get_models_from_env

# ---------------- CONFIGURATION ---------------- #

load_dotenv(override=True)

# Patch for https://github.com/VikParuchuri/marker/issues/442
torch.classes.__path__ = []

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


# Fix marker warning from torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

models = get_models_from_env(
    url_embedding="URL_EMBEDDING_MODEL", url_generative="URL_GENERATIVE_MODEL", url_reranking="URL_RERANKING_MODEL"
)
embedding_model = models.get("embedding")
generative_model = models.get("completion")
reranking_model = models.get("reranking")


# ---------------- INITIALIZATION ---------------- #


@st.cache_resource(show_spinner=False)
def initialize_clients_cache(config: dict, embedding_model=embedding_model, engine=ENGINE, **kwargs):
    return initialize_clients(config=config, embedding_model=embedding_model, engine=engine, **kwargs)


retriever, chat_client, qdrant_client = initialize_clients_cache(
    config=config,
    embedding_model=embedding_model,
    use_reranking=False,
    url_reranker=os.getenv("URL_RERANKING_MODEL"),
    model_reranker=models.get("reranking"),
)


# -----------------------------------


from pathlib import Path

from chatlas import ChatOpenAI
from dotenv import load_dotenv
from faicons import icon_svg
from shiny import App, Inputs, reactive, ui

welcome = """
Welcome to a choose-your-own learning adventure in data science!
Please pick your role, the size of the company you work for, and the industry you're in.
Then click the "Start adventure" button to begin.
"""


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize(
            "company_role",
            "You are a...",
            choices=[
                "Machine Learning Engineer",
                "Data Analyst",
                "Research Scientist",
                "MLOps Engineer",
                "Data Science Generalist",
            ],
            selected="Data Analyst",
        ),
        ui.input_selectize(
            "company_size",
            "who works for...",
            choices=[
                "yourself",
                "a startup",
                "a university",
                "a small business",
                "a medium-sized business",
                "a large business",
                "an enterprise corporation",
            ],
            selected="a medium-sized business",
        ),
        ui.input_selectize(
            "company_industry",
            "in the ... industry",
            choices=[
                "Healthcare and Pharmaceuticals",
                "Banking, Financial Services, and Insurance",
                "Technology and Software",
                "Retail and E-commerce",
                "Media and Entertainment",
                "Telecommunications",
                "Automotive and Manufacturing",
                "Energy and Oil & Gas",
                "Agriculture and Food Production",
                "Cybersecurity and Defense",
            ],
            selected="Healthcare and Pharmaceuticals",
        ),
        ui.input_action_button(
            "go",
            "Start adventure",
            icon=icon_svg("play"),
            class_="btn btn-primary",
        ),
        id="sidebar",
    ),
    ui.chat_ui("chat", messages=[welcome]),
    title="Choose your own data science adventure",
    fillable=True,
    fillable_mobile=True,
)


def server(input: Inputs):
    # Create a ChatAnthropic client with a system prompt
    app_dir = Path(__file__).parent
    with open(app_dir / "app-shiny/prompt.md") as f:
        system_prompt = f.read()

    _ = load_dotenv()
    chat_client = ChatOpenAI(
        system_prompt=system_prompt,
        base_url=config.get("OPENAI_API_BASE_GENERATIVE"),
        api_key="EMPTY",
        model=generative_model
    )

    chat = ui.Chat(id="chat")

    # The 'starting' user prompt is a function of the inputs
    @reactive.calc
    def starting_prompt():
        return (
            f"I want a story that features a {input.company_role()} "
            f"who works for {input.company_size()} in the {input.company_industry()} industry."
        )

    # Has the adventure started?
    has_started: reactive.value[bool] = reactive.value(False)

    # When the user clicks the 'go' button, start/restart the adventure
    @reactive.effect
    @reactive.event(input.go)
    async def _():
        if has_started():
            await chat.clear_messages()
            await chat.append_message(welcome)
        chat.update_user_input(value=starting_prompt(), submit=True)
        chat.update_user_input(value="", focus=True)
        has_started.set(True)

    @reactive.effect
    async def _():
        if has_started():
            ui.update_action_button(
                "go", label="Restart adventure", icon=icon_svg("repeat")
            )
            ui.update_sidebar("sidebar", show=False)
        else:
            chat.update_user_input(value=starting_prompt())

    @chat.on_user_submit
    async def _(user_input: str):
        n_msgs = len(chat.messages())
        if n_msgs == 1:
            user_input += " Please jump right into the story without any greetings or introductions."
        elif n_msgs == 4:
            user_input += ". Time to nudge this story toward its conclusion. Give one more scenario (like creating a report, dashboard, or presentation) that will let me wrap this up successfully."
        elif n_msgs == 5:
            user_input += ". Time to wrap this up. Conclude the story in the next step and offer to summarize the chat or create example scripts in R or Python. Consult your instructions for the correct format. If the user asks for code, remember that you'll need to create simulated data that matches the story."

        response = chat_client.stream(user_input)
        await chat.append_message_stream(response)


app = App(app_ui, server)
