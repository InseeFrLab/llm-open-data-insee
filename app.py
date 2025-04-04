import os
from datetime import datetime

import pandas as pd
import s3fs
import streamlit as st
import torch
from dotenv import load_dotenv
from loguru import logger

from src.app.feedbacks import feedback_titles, render_feedback_section
from src.app.history import activate_old_conversation, create_unique_id, summarize_conversation
from src.app.utils import generate_answer_from_context, initialize_clients
from src.config import set_config
from src.utils import create_prompt_from_instructions, question_instructions, system_instructions
from src.utils.utils_vllm import get_models_from_env
from src.vectordatabase.output_parsing import langchain_documents_to_df

# ---------------- CONFIGURATION ---------------- #

load_dotenv(override=True)

# Patch for https://github.com/VikParuchuri/marker/issues/442
torch.classes.__path__ = []

ENGINE = "chroma"

config = set_config(
    use_vault=True,
    components=["s3", "mlflow", "database", "model"],
    models_location={
        "url_embedding_model": "ENV_URL_EMBEDDING_MODEL",
        "url_generative_model": "ENV_URL_GENERATIVE_MODEL",
    },
    database_manager=ENGINE
    # override={"QDRANT_COLLECTION_NAME": "dirag_experimentation_d9867c0409cf44e1b222f9f5ede05c06"},
)

fs = s3fs.S3FileSystem(endpoint_url=config.get("endpoint_url"))
path_log = os.getenv("PATH_LOG_APP")


# Fix marker warning from torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

models = get_models_from_env(url_embedding="URL_EMBEDDING_MODEL", url_generative="URL_GENERATIVE_MODEL")
embedding_model = models.get("embedding")
generative_model = models.get("completion")


# ---------------- INITIALIZATION ---------------- #


@st.cache_resource(show_spinner=False)
def initialize_clients_cache(
    config: dict,
    embedding_model=embedding_model,
    engine=ENGINE
):
    return initialize_clients(config=config, embedding_model=embedding_model, engine=engine)


retriever, chat_client, qdrant_client = initialize_clients_cache(config=config, embedding_model=embedding_model)
n_docs = "XXXXX"
#n_docs = get_number_docs_collection(qdrant_client, config.get("QDRANT_COLLECTION_NAME"))
prompt = create_prompt_from_instructions(system_instructions, question_instructions)

# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="Chat with AI")

st.markdown(
    """
    <style>
        .st-emotion-cache-janbn0 {
            flex-direction: row-reverse;
            text-align: right;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- INITIALIZE SESSION STATES ---------------- #
DEFAULT_USERNAME = "anonymous"
st.session_state.setdefault("conversion_history", [])
st.session_state.setdefault("history", [])
st.session_state.setdefault("feedback", [])
st.session_state.setdefault("active_chat_history", None)
st.session_state.setdefault("clicked", False)
st.session_state.setdefault("username", DEFAULT_USERNAME)
st.session_state.setdefault("sidebar_conversations", None)
st.session_state.setdefault("just_loaded_history", False)
st.session_state.setdefault("has_initialized_conversation", False)

if "unique_id" not in st.session_state:
    st.session_state.unique_id = create_unique_id()

if st.session_state.active_chat_history is not None:
    st.session_state.unique_id = st.session_state.active_chat_history

unique_id = st.session_state.unique_id
active_user = st.session_state.username

# ---------------- SIDEBAR: HISTORY ---------------- #
sc1, sc2 = st.sidebar.columns((6, 1))

with st.sidebar:
    username = st.text_input("username", DEFAULT_USERNAME)

    if username != st.session_state.username:
        st.session_state.username = username
        st.session_state.sidebar_conversations = None
        st.session_state.just_loaded_history = False
        st.session_state.has_initialized_conversation = False

        st.session_state.unique_id = create_unique_id()
        st.session_state.history = []
        st.session_state.feedback = []
        st.session_state.active_chat_history = None

        st.rerun()

    if st.button("➕ Nouvelle conversation", key="new_convo"):
        st.session_state.unique_id = create_unique_id()
        st.session_state.history = [
            {
                "role": "assistant",
                "content": f"Bonjour ! Interrogez moi sur le site insee.fr ({n_docs} pages dans ma base de connaissance)",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "id": st.session_state.unique_id,
            }
        ]
        st.session_state.feedback = []
        st.session_state.active_chat_history = None
        st.session_state.has_initialized_conversation = True
        st.session_state.just_loaded_history = False
        st.rerun()

    if st.session_state.username == "anonymous":
        st.markdown("### To get a conversation history, change the username")
    else:
        st.markdown("### 💬 Past Conversations")

        if st.session_state.sidebar_conversations is None:
            try:
                directory = fs.ls(f"{path_log}/{username}/history")
                directory = [dir for dir in directory if dir.endswith(".parquet")]
            except FileNotFoundError:
                directory = []

            history_as_parquet = [pd.read_parquet(f, filesystem=fs) for f in directory]

            old_conversations = [
                summarize_conversation(chat_client, generative_model, history)
                for history in history_as_parquet
                if history is not None
            ]

            st.session_state.sidebar_conversations = old_conversations

            # ✅ Save sidebar conversations as a snapshot
            if old_conversations:
                df_conversations = pd.DataFrame(old_conversations)
                df_conversations["date"] = pd.to_datetime(df_conversations["date"], errors="coerce")
                df_conversations = df_conversations.sort_values(by="date", ascending=False)
                df_conversations.to_parquet(
                    f"{path_log}/{username}/conversation_history.parquet", index=False, filesystem=fs
                )

        for conversations in st.session_state.sidebar_conversations:
            convo_id = conversations["id"]
            title = conversations["summary"]

            is_active = st.session_state.active_chat_history == convo_id

            if is_active:
                st.markdown(
                    f"""
                    <div style='
                        background-color: #2e3a48;
                        padding: 0.5em;
                        border-radius: 0.5em;
                        margin-bottom: 0.3em;
                        color: white;
                        font-weight: bold;
                    '>
                        {title}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                if st.button(title, key=f"{convo_id}", on_click=activate_old_conversation, args=(convo_id, title)):
                    pass


# ---------------- INITIAL MESSAGE / LOAD HISTORY ---------------- #
if st.session_state.active_chat_history is not None and not st.session_state.just_loaded_history:
    id_unique = st.session_state.active_chat_history

    # Read and sort history
    history = pd.read_parquet(f"{path_log}/{username}/history/{id_unique}.parquet", filesystem=fs)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.sort_values(by="date")
    history["date"] = history["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Store back to session state
    st.session_state.history = history.to_dict(orient="records")

    st.session_state.unique_id = id_unique
    st.session_state.just_loaded_history = True
    st.rerun()

if not st.session_state.has_initialized_conversation and st.session_state.active_chat_history is None:
    st.session_state.history = [
        {
            "role": "assistant",
            "content": f"Bonjour ! Interrogez moi sur le site insee.fr ({n_docs} pages dans ma base de connaissance)",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id": st.session_state.unique_id,
        }
    ]
    st.session_state.has_initialized_conversation = True

# ---------------- CHAT MESSAGES & FEEDBACK ---------------- #
for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and i > 0:
            best_documents = retriever.invoke(st.session_state.history[i - 1]["content"])

            feedback_results = [
                render_feedback_section(
                    index=i,
                    message=message,
                    title=cfg["title"],
                    optional_text=cfg["optional_text"],
                    key_prefix=cfg["key_prefix"],
                    unique_id=unique_id,
                    feedback_type=cfg["feedback_type"],
                )
                for cfg in feedback_titles
            ]

        if len(st.session_state["history"]) > 1:
            conversation_history = pd.DataFrame(st.session_state["history"])
            feedback_history = pd.DataFrame(st.session_state["feedback"])
            conversation_history.to_parquet(
                f"{path_log}/{username}/history/{unique_id}.parquet", index=False, filesystem=fs
            )
            feedback_history.to_parquet(
                f"{path_log}/{username}/feedbacks/{unique_id}.parquet", index=False, filesystem=fs
            )

# ---------------- HANDLE USER INPUT ---------------- #
if user_query := st.chat_input("Poser une question sur le site insee"):
    st.session_state.history.append(
        {"role": "user", "content": user_query, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "id": unique_id}
    )

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
        }
    )

    st.rerun()
