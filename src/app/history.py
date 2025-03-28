import uuid

import streamlit as st

from src.app.utils import get_conversation_title


def create_unique_id() -> str:
    return str(uuid.uuid1())


def activate_old_conversation(convo_id, title):
    st.session_state.active_chat_history = convo_id
    st.session_state.active_chat_title = title
    st.session_state.just_loaded_history = False  # Mark it to load in the next run
    # No st.rerun() here


def summarize_conversation(chat_client, generative_model, history):
    if history is None:
        return None
    questions_asked = history.loc[history["role"] == "user"]["content"].tolist()
    questions_asked[0] = "Question: " + questions_asked[0]
    questions_asked = "\nQuestion: ".join(questions_asked)
    conversation_summary = get_conversation_title(chat_client, generative_model, questions_asked)
    return {"id": history["id"].iloc[0], "date": history["date"].iloc[0], "summary": conversation_summary}
