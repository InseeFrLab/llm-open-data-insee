from datetime import datetime

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from openai import OpenAI

from src.vectordatabase.chroma import chroma_vectorstore_as_retriever
from src.vectordatabase.client import create_client_and_collection
from src.vectordatabase.output_parsing import format_docs, langchain_documents_to_df
from src.vectordatabase.qdrant import qdrant_vectorstore_as_retriever
from src.vectordatabase.reranker import RerankerRetriever

with open("./prompt/system.md", encoding="utf-8") as f:
    system_instructions = f.read()

def initialize_clients(
    config: dict,
    embedding_model: str,
    number_retrieved_documents: str = 5,
    engine: str = "qdrant",
    use_reranking: bool = False,
    **kwargs,
):
    emb_model = OpenAIEmbeddings(
        model=embedding_model,
        base_url=config.get("OPENAI_API_BASE_EMBEDDING"),
        api_key=config.get("OPENAI_API_KEY_EMBEDDING"),
        tiktoken_enabled=False,
    )

    url_database_client = config.get(f"{engine.upper()}_URL")
    api_key_database_client = config.get(f"{engine.upper()}_API_KEY")
    collection_name = config.get(f"{engine.upper()}_COLLECTION_NAME")

    model_max_len = len(emb_model.embed_query("retrieving hidden_size"))
    # model_max_len = get_model_max_len(model_id=emb_model.model)

    client = create_client_and_collection(
        url=url_database_client,
        api_key=api_key_database_client,
        collection_name=collection_name,
        model_max_len=model_max_len,
        engine=engine,
        vector_name=embedding_model,
    )

    constructor_retriever = qdrant_vectorstore_as_retriever
    if engine == "chroma":
        constructor_retriever = chroma_vectorstore_as_retriever

    retriever = constructor_retriever(
        client=client,
        collection_name=collection_name,
        embedding_function=emb_model,
        vector_name=emb_model.model,
        number_retrieved_docs=number_retrieved_documents,
    )

    logger.success("Retriever initialized successfully")

    if use_reranking is True:
        if "url_reranker" not in kwargs or "model_reranker" not in kwargs:
            raise ValueError("url_reranker and model_reranker need to be provided if use_reranking is True")
        retriever = RerankerRetriever(retriever=retriever, **kwargs)

    chat_client = OpenAI(
        base_url=config.get("OPENAI_API_BASE_GENERATIVE"),
        api_key=config.get("OPENAI_API_KEY_GENERATIVE"),
    )
    return retriever, chat_client


def generate_answer_from_context(retriever, chat_client, generative_model: str, prompt: str, question: str):
    best_documents = retriever.invoke(question)
    best_documents_df = langchain_documents_to_df(best_documents)
    logger.debug(best_documents_df)
    context = format_docs(best_documents)
    question_with_context = prompt.format(question=question, context=context)

    logger.debug(question_with_context)

    stream = chat_client.chat.completions.create(
        model=generative_model,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": question_with_context},
        ],
        stream=True,
    )
    return stream


def create_assistant_message(content: str = None, role: str = "assistant") -> dict:
    return {
        "role": role,
        "content": content,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id": st.session_state.unique_id,
    }
