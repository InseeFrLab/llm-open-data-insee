import uuid

import pandas as pd
import s3fs
import streamlit as st
from langchain_core.prompts import PromptTemplate

with open("./prompt/summarizer_question.md", encoding="utf-8") as f:
    question_summarizer = f.read()

with open("./prompt/summarizer_system.md", encoding="utf-8") as f:
    system_summarizer = f.read()


def create_unique_id() -> str:
    return str(uuid.uuid1())


def activate_old_conversation(convo_id, title):
    st.session_state.active_chat_history = convo_id
    st.session_state.active_chat_title = title
    st.session_state.just_loaded_history = False  # Mark it to load in the next run


# READ/WRITE HISTORY ----------------------------------------


def read_history_from_parquet(path_log: str, username: str, filesystem: s3fs.S3FileSystem):
    try:
        directory = filesystem.ls(f"{path_log}/{username}/history")
        directory = [dir for dir in directory if dir.endswith(".parquet")]
    except FileNotFoundError:
        directory = []
        return

    history_as_parquet = [pd.read_parquet(f, filesystem=filesystem) for f in directory]
    return history_as_parquet


def snapshot_sidebar_conversations(
    old_conversations: dict, path_log: str, username: str, filesystem: s3fs.S3FileSystem
):
    df_conversations = pd.DataFrame(old_conversations)
    df_conversations = _format_history_df(df_conversations)
    df_conversations.to_parquet(
        f"{path_log}/{username}/conversation_history.parquet", index=False, filesystem=filesystem
    )


def _format_history_df(history):
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["date"] = history["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    history = history.sort_values(by="date")
    return history


def restore_history(path_log: str, username: str, id_unique: str, filesystem: s3fs.S3FileSystem):
    history = pd.read_parquet(f"{path_log}/{username}/history/{id_unique}.parquet", filesystem=filesystem)
    history = _format_history_df(history)
    return history


# GET CONVERSATION ----------------------------------------


def get_conversation_title(chat_client, generative_model, full_text, instructions: dict = None):
    if instructions is None:
        instructions = {"system": system_summarizer, "user": question_summarizer}

    prompt_summarizer = PromptTemplate.from_template(question_summarizer)

    prompt_summarizer = prompt_summarizer.format(conversation=full_text)

    response = chat_client.chat.completions.create(
        model=generative_model,
        messages=[{"role": "user", "content": prompt_summarizer}],
        stop=None,
    )
    conversation_title = response.choices[0].message.content

    return conversation_title


def summarize_conversation(chat_client, generative_model, history):
    if history is None:
        return None

    questions_asked = history.loc[history["role"] == "user"]["content"].tolist()
    questions_asked[0] = "Question: " + questions_asked[0]
    questions_asked = "\nQuestion: ".join(questions_asked)
    conversation_summary = get_conversation_title(chat_client, generative_model, questions_asked)
    return {"id": history["id"].iloc[0], "date": history["date"].iloc[0], "summary": conversation_summary}
