import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from qdrant_client import QdrantClient
from openai import OpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from streamlit_feedback import streamlit_feedback

from src.utils import create_prompt_from_instructions, format_docs
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

config = {
    **config_s3, **config_database_client, **config_embedding_model, **config_generative_model
}

embedding_model = get_model_from_env("URL_EMBEDDING_MODEL")
generative_model = get_model_from_env("URL_GENERATIVE_MODEL")


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

retriever, chat_client, qdrant_client = initialize_clients(config)


def get_number_docs_collection(qdrant_client, collection_name=config.get("QDRANT_COLLECTION_NAME")):
    collection_info = qdrant_client.get_collection(collection_name=collection_name)
    return collection_info.points_count

n_docs = get_number_docs_collection(qdrant_client, config.get("QDRANT_COLLECTION_NAME"))


# ---------------- PROMPT TEMPLATE ---------------- #
system_instructions = """
Tu es un assistant spécialisé dans la statistique publique.
Tu réponds à des questions concernant les données de l'Insee, l'institut national statistique Français.

Réponds en FRANCAIS UNIQUEMENT. Utilise une mise en forme au format markdown.

En utilisant UNIQUEMENT les informations présentes dans le contexte, réponds de manière argumentée à la question posée.

Cite 5 sources maximum et mentionne l'url d'origine.

Si tu ne peux pas induire ta réponse du contexte, ne réponds pas.

Voici le contexte sur lequel tu dois baser ta réponse :
Contexte: {context}
"""

question_instructions = """
Voici la question à laquelle tu dois répondre :
Question: {question}

Réponse:
"""

prompt = create_prompt_from_instructions(system_instructions, question_instructions)


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


# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="Chat with AI")

if "history" not in st.session_state:
    st.session_state.history = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

def handle_feedback(response, index):
    st.toast("✔️ Feedback received!")
    st.session_state.feedback.append({"index": index, "response": response})
    logger.debug(st.session_state)


initial_message = f"Interrogez moi sur le site insee.fr ({n_docs} pages dans ma base de connaissance)"

if not st.session_state.history:
    st.session_state.history.append({"role": "assistant", "content": initial_message})

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            feedback = streamlit_feedback(
                on_submit=lambda response, idx=i: handle_feedback(response, idx),
                feedback_type="faces",
                optional_text_label="Qualité du retriever",
                key=f"feedback_{i}"
            )

if user_query := st.chat_input("Poser une question sur le site insee"):
    st.session_state.history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    with st.chat_message("assistant"):
        response = st.write_stream(generate_answer_from_context(retriever, prompt, user_query))
    st.session_state.history.append({"role": "assistant", "content": response})
    st.rerun()