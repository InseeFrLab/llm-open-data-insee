import os
import pathlib
from datetime import datetime
import uuid

from dotenv import load_dotenv
from loguru import logger
from pathlib import Path

import pandas as pd
import streamlit as st
import torch


from src.app.utils import create_config_app, generate_answer_from_context, initialize_clients
from src.app.feedbacks import feedback_titles, render_feedback_section

from src.db_building.get_number_documents import get_number_docs_collection
from src.utils import create_prompt_from_instructions, question_instructions, system_instructions
from src.utils.utils_vllm import get_model_from_env

# ---------------- CONFIGURATION ---------------- #

# Load environment variables
load_dotenv(override=True)
config = create_config_app()

# Fix weird warning (https://github.com/VikParuchuri/marker/issues/442)
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

embedding_model = get_model_from_env("URL_EMBEDDING_MODEL")
generative_model = get_model_from_env("URL_GENERATIVE_MODEL")
logger.debug(f"Embedding model used: {embedding_model}")
logger.debug(f"Generative model used: {generative_model}")

#@st.cache_resource(show_spinner=False)
def create_unique_id():
    return str(uuid.uuid1())

unique_id = create_unique_id()

# ---------------- INITIALIZATION ---------------- #


@st.cache_resource(show_spinner=False)
def initialize_clients_cache(config: dict, embedding_model=embedding_model):
    return initialize_clients(config=config, embedding_model=embedding_model)


retriever, chat_client, qdrant_client = initialize_clients_cache(config=config, embedding_model=embedding_model)
n_docs = get_number_docs_collection(qdrant_client, config.get("QDRANT_COLLECTION_NAME"))

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

if "history" not in st.session_state:
    st.session_state.history = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []


# LEFT SIDEBAR: CONVERSATION HISTORY --------------

st.sidebar.header("💬 Past Conversations")

sc1, sc2 = st.sidebar.columns((6, 1))

from src.app.utils import get_conversation_title

def summarize_conversation(history):
    
    if history is None:
        exit

    questions_asked = history.loc[history["role"] == "user"]["content"]
    questions_asked = "\n".join(questions_asked)

    conversation_summary = get_conversation_title(
            chat_client, generative_model, questions_asked
    )

    return {"id": history["id"].iloc[0], "date": history["date"].iloc[0], "summary": conversation_summary}


with st.sidebar:
    username = st.text_input("username", "anonymous")
    pathlib.Path(f"./logs/{username}/history").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"./logs/{username}/feedbacks").mkdir(parents=True, exist_ok=True)
    directory = Path(f"./logs/{username}/history").glob("*.parquet")
    history_as_parquet = list(directory)
    history_as_parquet = [pd.read_parquet(f) for f in history_as_parquet]
    old_conversations = [summarize_conversation(history) for history in history_as_parquet]
    for conversations in old_conversations:
        key=conversations["id"]
        title=conversations["summary"]
        if sc1.button(title, key=f"c{key}"):
            st.sidebar.info(f'{title}', icon="💬")

        if sc2.button("❌", key=f"x{key}"):
            st.sidebar.info("Conversation removed", icon="❌")
            #cookie_manager.delete(conversation_id)        




# FEEDBACK RELATED STUFF ------------------

initial_message = f"Interrogez moi sur le site insee.fr ({n_docs} pages dans ma base de connaissance)"

if not st.session_state.history:
    st.session_state.history.append(
        {
            "role": "assistant", "content": initial_message,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id": unique_id
        }
    )

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and i > 0:
            best_documents = retriever.invoke(st.session_state.history[i - 1]["content"])
            best_documents_df = [docs.metadata for docs in best_documents]
            best_documents_df = pd.DataFrame(best_documents_df)
            # stoggle(
            # "Documents renvoyés",
            # st.write(best_documents_df),
            # )
            feedback_results = []
            for cfg in feedback_titles:
                feedback_results.append(
                    render_feedback_section(
                        index=i,
                        message=message,
                        title=cfg["title"],
                        optional_text=cfg["optional_text"],
                        key_prefix=cfg["key_prefix"],
                        unique_id=unique_id,
                        feedback_type=cfg["feedback_type"],
                    )
                )

        if len(st.session_state["history"])>1:
            conversation_history = pd.DataFrame(st.session_state["history"])
            feedback_history = pd.DataFrame(st.session_state["feedback"])
            conversation_history.to_parquet(f"logs/{username}/history/{unique_id}.parquet")
            feedback_history.to_parquet(f"logs/{username}/feedbacks/{unique_id}.parquet")
        # st.write(conversation_history)
        # st.write(feedback_history)


if user_query := st.chat_input("Poser une question sur le site insee"):
    st.session_state.history.append(
        {
            "role": "user", "content": user_query, 
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id": unique_id
        }
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
        {"role": "assistant", "content": response,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id": unique_id}
    )
    st.rerun()


