import os
from datetime import datetime
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from loguru import logger

from langfuse import Langfuse
from langfuse.decorators import observe
from langfuse.openai import OpenAI

from src.vectordatabase.chroma import chroma_vectorstore_as_retriever
from src.vectordatabase.client import create_client_and_collection
from src.vectordatabase.output_parsing import format_docs, langchain_documents_to_df
from src.vectordatabase.qdrant import qdrant_vectorstore_as_retriever
from src.vectordatabase.reranker import RerankerRetriever


langfuse = Langfuse()
system_prompt = langfuse.get_prompt("system_prompt", label="latest")
user_prompt = langfuse.get_prompt("user_prompt", label="latest")


def initialize_clients(
    config: dict,
    embedding_model: str,
    number_retrieved_documents: str = 5,
    engine: str = "qdrant",
    use_reranking: bool = False,
    emb_model_client=None,
    **kwargs,
):
    if emb_model_client is None:
        emb_model = OpenAIEmbeddings(
            model=embedding_model,
            base_url=config.get("OPENAI_API_BASE_EMBEDDING"),
            api_key=config.get("OPENAI_API_KEY_EMBEDDING"),
            tiktoken_enabled=False,
        )
    else:
        emb_model = emb_model_client

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
            raise ValueError(
                "url_reranker and model_reranker need to be provided if use_reranking is True"
            )
        retriever = RerankerRetriever(retriever=retriever, **kwargs)

    chat_client = OpenAI(
        base_url=config.get("OPENAI_API_BASE_GENERATIVE"),
        api_key=config.get("OPENAI_API_KEY_GENERATIVE"),
    )
    return retriever, chat_client

@observe()
def generate_answer_from_context(
    retriever, chat_client,
    generative_model: str,
    question: str
):
    best_documents = retriever.invoke(question)
    best_documents_df = langchain_documents_to_df(best_documents)
    logger.debug(best_documents_df)
    context = format_docs(best_documents)
    question_with_context = user_prompt.compile(question=question, context=context)

    logger.debug(question_with_context)

    stream = chat_client.chat.completions.create(
        name="query_generative_app",
        model=generative_model,
        messages=[
            {"role": "system", "content": system_prompt.compile() },
            {"role": "user", "content": question_with_context},
        ],
        stream=True,
        langfuse_prompt=system_prompt
    )
    return stream


def create_assistant_message(content: str = None, role: str = "assistant", unique_id: str = None) -> dict:
    return {
        "role": role,
        "content": content,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id": unique_id,
    }


def flatten_history_for_parquet(history):
    flat_history = []
    for message in history:
        flat = {
            "role": message.get("role"),
            "content": message.get("content"),
            "date": message.get("date"),
            "id": message.get("id"),
            "collection": message.get("collection"),
        }
        flat_history.append(flat)
    return pd.DataFrame(flat_history)
