import os
import pathlib
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from openai import OpenAI
from qdrant_client import QdrantClient
from streamlit_feedback import streamlit_feedback

from src.utils import create_prompt_from_instructions, format_docs, question_instructions, system_instructions
from src.utils.utils_vllm import get_model_from_env

# Load environment variables
load_dotenv(override=True)

# ---------------- CONFIGURATION ---------------- #
config_s3 = {
    "AWS_ENDPOINT_URL": os.getenv("AWS_ENDPOINT_URL", "https://minio.lab.sspcloud.fr"),
}

config_database_client = {
    "QDRANT_URL": os.getenv("QDRANT_URL"),
    "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
    "QDRANT_COLLECTION_NAME": os.getenv("COLLECTION_NAME", "dirag_mistral_small"),
}

config_embedding_model = {
    "OPENAI_API_BASE_EMBEDDING": os.getenv("OPENAI_API_BASE", os.getenv("URL_EMBEDDING_MODEL")),
    "OPENAI_API_KEY_EMBEDDING": os.getenv("OPENAI_API_KEY", "EMPTY"),
}

config_generative_model = {
    "OPENAI_API_BASE_GENERATIVE": os.getenv("OPENAI_API_BASE", os.getenv("URL_GENERATIVE_MODEL")),
    "OPENAI_API_KEY_GENERATIVE": os.getenv("OPENAI_API_KEY", "EMPTY"),
}

config = {**config_s3, **config_database_client, **config_embedding_model, **config_generative_model}

embedding_model = get_model_from_env("URL_EMBEDDING_MODEL")
generative_model = get_model_from_env("URL_GENERATIVE_MODEL")
logger.debug(f"Embedding model used: {embedding_model}")
logger.debug(f"Generative model used: {generative_model}")

unique_id = str(uuid.uuid1())
pathlib.Path("./logs/history").mkdir(parents=True, exist_ok=True)
pathlib.Path("./logs/feedbacks").mkdir(parents=True, exist_ok=True)


# ---------------- INITIALIZATION ---------------- #
@st.cache_resource(show_spinner=False)
def initialize_clients(config):
    emb_model = OpenAIEmbeddings(
        model=embedding_model,
        base_url=config.get("OPENAI_API_BASE_EMBEDDING"),
        api_key=config.get("OPENAI_API_KEY_EMBEDDING"),
    )

    qdrant_client = QdrantClient(
        url=config.get("QDRANT_URL"), api_key=config.get("QDRANT_API_KEY"), port="443", https=True
    )
    logger.success("Connected to Qdrant DB client")

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=config.get("QDRANT_COLLECTION_NAME"),
        embedding=emb_model,
        vector_name=embedding_model,
    )

    logger.success("Vectorstore initialized successfully")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    chat_client = OpenAI(
        base_url=config.get("OPENAI_API_BASE_GENERATIVE"),
        api_key=config.get("OPENAI_API_KEY_GENERATIVE"),
    )
    return retriever, chat_client, qdrant_client


def get_number_docs_collection(qdrant_client, collection_name=config.get("QDRANT_COLLECTION_NAME")):
    collection_info = qdrant_client.get_collection(collection_name=collection_name)
    return collection_info.points_count


def generate_answer_from_context(retriever, prompt, question):
    best_documents = retriever.invoke(question)
    context = format_docs(best_documents)
    question_with_context = prompt.format(question=question, context=context)

    stream = chat_client.chat.completions.create(
        model=generative_model,
        messages=[{"role": "user", "content": question_with_context}],
        stream=True,
    )
    return stream


retriever, chat_client, qdrant_client = initialize_clients(config)
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


# FEEDBACK RELATED STUFF ------------------

css_annotation_title = "text-align: right; font-weight: bold; font-style: italic;"


def handle_feedback(response, index, history, feedback_type="retriever"):
    st.toast("✔️ Feedback received!")
    message = history[index]["content"]
    question = history[index - 1]["content"]
    submission_time = datetime.now().strftime("%H:%M:%S")  # Only the time as string

    st.session_state.feedback.append(
        {
            "discussion_index": index,
            "evaluation": response["score"],
            "evaluation_text": response["text"],
            "question": question,
            "answer": message,
            "type": feedback_type,
            "submitted_at": submission_time,
            "unique_id": unique_id,
        }
    )
    # st.write(st.session_state.feedback)


def render_feedback_section(index, message, title, optional_text, key_prefix, feedback_type):
    with st.container(key=f"{key_prefix}-{index}"):
        st.markdown(f"<p style='{css_annotation_title}'>{title}</p>", unsafe_allow_html=True)
        return streamlit_feedback(
            on_submit=lambda response, idx=index, msg=message: handle_feedback(
                response, idx, st.session_state.history, feedback_type=feedback_type
            ),
            feedback_type="faces",
            optional_text_label=optional_text,
            key=f"{key_prefix}_{index}",
        )


feedback_titles = [
    {
        "title": "Evaluation de la pertinence des documents renvoyés",
        "optional_text": "Pertinence des documents",
        "key_prefix": "feedback-retriever",
        "feedback_type": "retriever",
    },
    {
        "title": "Evaluation de la qualité de la réponse à l'aune du contexte fourni (fond):",
        "optional_text": "Qualité du fond",
        "key_prefix": "feedback-generation",
        "feedback_type": "generation_fond",
    },
    {
        "title": "Evaluation de la qualité de la réponse (style, mise en forme, etc.):",
        "optional_text": "Qualité de la forme",
        "key_prefix": "feedback-generation-mef",
        "feedback_type": "generation_forme",
    },
]

initial_message = f"Interrogez moi sur le site insee.fr ({n_docs} pages dans ma base de connaissance)"

if not st.session_state.history:
    st.session_state.history.append({"role": "assistant", "content": initial_message})

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if i > 0:
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
                            feedback_type=cfg["feedback_type"],
                        )
                    )

        conversation_history = pd.DataFrame(st.session_state["history"])
        feedback_history = pd.DataFrame(st.session_state["feedback"])
        conversation_history.to_parquet(f"logs/history/{unique_id}.parquet")
        feedback_history.to_parquet(f"logs/feedbacks/{unique_id}.parquet")
        # st.write(conversation_history)
        # st.write(feedback_history)


if user_query := st.chat_input("Poser une question sur le site insee"):
    st.session_state.history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    with st.chat_message("assistant"):
        response = st.write_stream(generate_answer_from_context(retriever, prompt, user_query))
    st.session_state.history.append({"role": "assistant", "content": response})
    # with open()
    st.rerun()
