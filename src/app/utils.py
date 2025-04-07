from langchain_openai import OpenAIEmbeddings
from loguru import logger
from openai import OpenAI
from langchain_core.prompts import PromptTemplate

from src.vectordatabase.output_parsing import format_docs
from src.model.prompt import (
    system_instructions,
    question_instructions_summarizer, system_instructions_summarizer
)

from src.vectordatabase.client import create_client_and_collection
from src.vectordatabase.qdrant import (
    qdrant_vectorstore_as_retriever
)
from src.vectordatabase.chroma import (
    chroma_vectorstore_as_retriever
)
from src.utils.utils_vllm import get_model_max_len
from src.vectordatabase.output_parsing import langchain_documents_to_df


def initialize_clients(
    config: dict, embedding_model: str,
    number_retrieved_documents: str = 5,
    engine: str = "qdrant"
):

    emb_model = OpenAIEmbeddings(
        model=embedding_model,
        base_url=config.get("OPENAI_API_BASE_EMBEDDING"),
        api_key=config.get("OPENAI_API_KEY_EMBEDDING"),
    )

    url_database_client = config.get(f"{engine.upper()}_URL")
    api_key_database_client = config.get(f"{engine.upper()}_API_KEY")
    collection_name = config.get(f"{engine.upper()}_COLLECTION_NAME")

    model_max_len = len(
            emb_model.embed_query("retrieving hidden_size")
        )
    # model_max_len = get_model_max_len(model_id=emb_model.model)

    client = create_client_and_collection(
            url=url_database_client,
            api_key=api_key_database_client,
            collection_name=collection_name,
            model_max_len=model_max_len,
            engine=engine,
            vector_name=embedding_model
        )

    constructor_retriever = qdrant_vectorstore_as_retriever
    if engine == "chroma":
        constructor_retriever = chroma_vectorstore_as_retriever

    retriever = constructor_retriever(
            client=client,
            collection_name=collection_name,
            embedding_function=emb_model,
            vector_name=emb_model.model,
            number_retrieved_docs=number_retrieved_documents
        )

    logger.success("Vectorstore initialized successfully")

    chat_client = OpenAI(
        base_url=config.get("OPENAI_API_BASE_GENERATIVE"),
        api_key=config.get("OPENAI_API_KEY_GENERATIVE"),
    )
    return retriever, chat_client, retriever


def get_conversation_title(chat_client, generative_model, full_text):

    prompt_summarizer = PromptTemplate.from_template(
        question_instructions_summarizer
    )

    prompt_summarizer = prompt_summarizer.format(conversation=full_text)

    response = chat_client.chat.completions.create(
        model=generative_model,
        messages=[{"role": "user", "content": prompt_summarizer}],
        stop=None,
    )
    conversation_title = response.choices[0].message.content

    return conversation_title


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
            {"role": "user", "content": question_with_context}
            ],
        stream=True,
    )
    return stream
