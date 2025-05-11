import os
import sys
import argparse
import tomllib
from datetime import datetime

import pandas as pd
import s3fs
import streamlit as st
import torch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from loguru import logger

from src.app.feedbacks import feedback_titles, render_feedback_section
from src.app.history import (
    activate_old_conversation,
    create_unique_id,
    read_history_from_parquet,
    restore_history,
    snapshot_sidebar_conversations,
    summarize_conversation,
)
from src.app.session import initialize_session_state, reset_session_state
from src.app.utils import create_assistant_message, generate_answer_from_context, initialize_clients, flatten_history_for_parquet
from src.config import set_config
from src.utils.utils_vllm import get_models_from_env

# ---------------- COMMAND-LINE ARGUMENTS ---------------- #

parser = argparse.ArgumentParser()
parser.add_argument("--log", action="store_true", help="Enable session state logging")
args, _ = parser.parse_known_args()
LOG_SESSION_STATE = args.log

def log_session_state(msg="Session State"):
    if LOG_SESSION_STATE:
        logger.info(f"{msg}: {dict(st.session_state)}")

if args.log:
    logger.add("log.txt", rotation="1 MB", enqueue=True, backtrace=True, diagnose=True)


# ---------------- CONFIGURATION ---------------- #

load_dotenv(override=True)
torch.classes.__path__ = []

ENGINE = "chroma"
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
)

fs = s3fs.S3FileSystem(endpoint_url=config.get("endpoint_url"))
path_log = os.getenv("PATH_LOG_APP")
collection_name = config.get(f"{ENGINE.upper()}_COLLECTION_NAME")

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

models = get_models_from_env(
    url_embedding="URL_EMBEDDING_MODEL", url_generative="URL_GENERATIVE_MODEL", url_reranking="URL_RERANKING_MODEL"
)
embedding_model = models.get("embedding")
generative_model = models.get("completion")
reranking_model = models.get("reranking")

# ---------------- INITIALIZATION ---------------- #

DEFAULT_USERNAME = "anonymous"
with open("./src/app/constants.toml", "rb") as f:
    messages = tomllib.load(f)
with open("./prompt/question.md", encoding="utf-8") as f:
    question_prompt = f.read()

@st.cache_resource(show_spinner=False)
def initialize_clients_cache(config: dict, embedding_model=embedding_model, engine=ENGINE, **kwargs):
    return initialize_clients(config=config, embedding_model=embedding_model, engine=engine, **kwargs)

retriever, chat_client = initialize_clients_cache(
    config=config,
    embedding_model=embedding_model,
    use_reranking=False,
    url_reranker=os.getenv("URL_RERANKING_MODEL"),
    model_reranker=models.get("reranking"),
)

prompt = PromptTemplate.from_template(question_prompt)

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="insee.fr assistant")

with open("./src/app/style.css") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ---------------- INITIALIZE SESSION STATES ---------------- #

initialize_session_state(
    {
        "conversion_history": [],
        "history": [],
        "feedback": [],
        "active_chat_history": None,
        "clicked": False,
        "username": DEFAULT_USERNAME,
        "sidebar_conversations": None,
        "just_loaded_history": False,
        "has_initialized_conversation": False,
        "retriever": [],
    }
)
log_session_state("After initialize_session_state")

if "unique_id" not in st.session_state:
    st.session_state.unique_id = create_unique_id()
    log_session_state("Generated unique_id")

if st.session_state.active_chat_history is not None:
    st.session_state.unique_id = st.session_state.active_chat_history
    log_session_state("Restored active_chat_history")

unique_id = st.session_state.unique_id
active_user = st.session_state.username

# ---------------- SIDEBAR: HISTORY ---------------- #

sc1, sc2 = st.sidebar.columns((6, 1))

with st.sidebar:
    username = st.text_input("username", DEFAULT_USERNAME)

    if username != st.session_state.username:
        reset_session_state(
            {
                "username": username,
                "sidebar_conversations": None,
                "just_loaded_history": False,
                "has_initialized_conversation": False,
                "unique_id": create_unique_id,
                "history": [],
                "feedback": [],
                "active_chat_history": None,
            }
        )
        log_session_state("After username change")
        st.rerun()

    if st.button("➕ Nouvelle conversation", key="new_convo"):
        start_message = create_assistant_message(content=messages["MESSAGE_START"], unique_id=unique_id)

        reset_session_state(
            {
                "unique_id": create_unique_id,
                "history": [start_message],
                "feedback": [],
                "active_chat_history": None,
                "has_initialized_conversation": True,
                "just_loaded_history": False,
            }
        )
        log_session_state("New conversation started")
        st.rerun()

    if st.session_state.username == "anonymous":
        st.markdown(messages["MESSAGE_PAST_CONVERSATION_ANON"])
    else:
        st.markdown(messages["MESSAGE_PAST_CONVERSATION"])

        if st.session_state.sidebar_conversations is None:
            history_as_parquet = read_history_from_parquet(path_log, username, fs)

            old_conversations = [
                summarize_conversation(chat_client, generative_model, history)
                for history in history_as_parquet
                if history is not None
            ]

            st.session_state.sidebar_conversations = old_conversations
            log_session_state("Loaded sidebar conversations")

            if old_conversations:
                snapshot_sidebar_conversations(
                    old_conversations=old_conversations, path_log=path_log, username=username, filesystem=fs
                )

        for conversations in st.session_state.sidebar_conversations:
            convo_id = conversations["id"]
            title = conversations["summary"]

            is_active = st.session_state.active_chat_history == convo_id

            if is_active:
                st.markdown(
                    f'<div class="active-conversation">{title}</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(title, key=f"{convo_id}", on_click=activate_old_conversation, args=(convo_id, title)):
                    log_session_state(f"Conversation selected: {convo_id}")

# ---------------- INITIAL MESSAGE / LOAD HISTORY ---------------- #

if st.session_state.active_chat_history is not None and not st.session_state.just_loaded_history:
    id_unique = st.session_state.active_chat_history
    history = restore_history(path_log, username, id_unique, filesystem=fs)

    reset_session_state(
        {
            "history": lambda: history.to_dict(orient="records"),
            "unique_id": id_unique,
            "just_loaded_history": True,
        }
    )
    log_session_state("History restored")
    st.rerun()

if not st.session_state.has_initialized_conversation and st.session_state.active_chat_history is None:
    st.session_state.history = [create_assistant_message(content=messages["MESSAGE_START"], unique_id=unique_id)]
    st.session_state.has_initialized_conversation = True
    log_session_state("Session initialized with welcome message")

# ---------------- CHAT MESSAGES & FEEDBACK ---------------- #

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and i > 0:
            best_documents = retriever.invoke(st.session_state.history[i - 1]["content"])
            st.session_state.retriever.append(best_documents)

            feedback_results = [
                render_feedback_section(
                    index=i,
                    message=message,
                    title=cfg["title"],
                    optional_text=cfg["optional_text"],
                    key_prefix=cfg["key_prefix"],
                    unique_id=unique_id,
                    db_collection=collection_name,
                    feedback_type=cfg["feedback_type"],
                )
                for cfg in feedback_titles
            ]

        if len(st.session_state["history"]) > 1:
            conversation_history = flatten_history_for_parquet(st.session_state["history"])
            feedback_history = pd.DataFrame(st.session_state["feedback"])
            logger.debug(conversation_history)

# ---------------- HANDLE USER INPUT ---------------- #

if user_query := st.chat_input("Poser une question sur le site insee"):
    st.session_state.history.append(
        {
            "role": "user",
            "content": user_query,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id": unique_id,
            "collection": collection_name
        }
    )
    log_session_state("After user query")

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response = st.write_stream(
            generate_answer_from_context(
                retriever=retriever,
                chat_client=chat_client,
                generative_model=generative_model,
                prompt=prompt,
                question=user_query,
            )
        )

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": response,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id": unique_id,
            "collection": collection_name
        }
    )
    log_session_state("After assistant response")

    st.rerun()
