
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from openai import OpenAI
from qdrant_client import QdrantClient

from src.utils import create_prompt_from_instructions, format_docs
from src.utils.prompt import question_instructions_summarizer, system_instructions_summarizer


def initialize_clients(config: dict, embedding_model: str):
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


def get_conversation_title(chat_client, generative_model, full_text):
    prompt_summarizer = create_prompt_from_instructions(
        system_instructions_summarizer, question_instructions_summarizer
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
    context = format_docs(best_documents)
    question_with_context = prompt.format(question=question, context=context)

    stream = chat_client.chat.completions.create(
        model=generative_model,
        messages=[{"role": "user", "content": question_with_context}],
        stream=True,
    )
    return stream
