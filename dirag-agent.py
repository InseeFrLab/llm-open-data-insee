import os

import chainlit as cl
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from src.model_building import cache_model_from_hf_hub
from src.utils import create_prompt_from_instructions, format_docs
from src.utils.utils_vllm import get_model_from_env

# CONFIGURATION ------------------------------------------

config_s3 = {"AWS_ENDPOINT_URL": os.getenv("AWS_ENDPOINT_URL", "https://minio.lab.sspcloud.fr")}

config_database_client = {
    "QDRANT_URL": os.getenv("QDRANT_URL", None),
    "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", None),
    "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "dirag_mistral_small"),
}

config_embedding_model = {
    # Assuming an OpenAI compatible client is used (VLLM, Ollama, etc.)
    "OPENAI_API_BASE_EMBEDDING": os.getenv("OPENAI_API_BASE", os.getenv("URL_EMBEDDING_MODEL")),
    "OPENAI_API_KEY_EMBEDDING": os.getenv("OPENAI_API_KEY", "EMPTY"),
    "MODE_EMBEDDING": "API",  # 'API' or 'local' mode
}

config_generative_model = {
    # Assuming an OpenAI compatible client is used (VLLM, Ollama, etc.)
    "OPENAI_API_BASE_GENERATIVE": os.getenv("OPENAI_API_BASE", os.getenv("URL_GENERATIVE_MODEL")),
    "OPENAI_API_KEY_GENERATIVE": os.getenv("OPENAI_API_KEY", "EMPTY"),
    "MODE_COMPLETION": "API",  # 'API' or 'local' mode
}


config = {**config_s3, **config_database_client, **config_embedding_model, **config_generative_model}


# ENVIRONEMENT ------------------------------

load_dotenv(override=True)

# TODO: alternative with local models
embedding_model = get_model_from_env("URL_EMBEDDING_MODEL")
generative_model = get_model_from_env("URL_GENERATIVE_MODEL")


vllm_client_completion = AsyncOpenAI(
    base_url=config.get("OPENAI_API_BASE_GENERATIVE"),
    api_key=config.get("OPENAI_API_KEY_GENERATIVE"),
)

# Instrument the OpenAI client
cl.instrument_openai()

settings = {
    "model": "mistralai/Mistral-Small-24B-Instruct-2501",
    "temperature": 0,
    # ... more settings
}


# PROMPT -------------------------------------

system_instructions = """
Tu es un assistant spécialisé dans la statistique publique.
Tu réponds à des questions concernant les données de l'Insee, l'institut national statistique Français.

Réponds en FRANCAIS UNIQUEMENT. Utilise une mise en forme au format markdown.

En utilisant UNIQUEMENT les informations présentes dans le contexte, réponds de manière argumentée à la question posée.

La réponse doit être développée et citer ses sources (titre et url de la publication) qui sont référencées à la fin.
Cite notamment l'url d'origine de la publication, dans un format markdown.

Cite 5 sources maximum.

Tu n'es pas obligé d'utiliser les sources les moins pertinentes.

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


def _embedding_client_local(config):
    cache_model_from_hf_hub(embedding_model, hf_token=os.getenv("HF_TOKEN"))

    emb_model = HuggingFaceEmbeddings(  # load from sentence transformers
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        show_progress=False,
    )

    return emb_model


def _embedding_client_api(config):
    emb_model = OpenAIEmbeddings(
        model=embedding_model,
        base_url=config.get("OPENAI_API_BASE_EMBEDDING"),
        api_key=config.get("OPENAI_API_KEY_EMBEDDING"),
    )

    return emb_model


@cl.cache
def load_retriever_cache():
    logger.info("Loading vector database")

    emb_model = _embedding_client_api()

    client = QdrantClient(url=config.get("QDRANT_URL"), api_key=config.get("QDRANT_API_KEY"), port="443", https="true")
    logger.success("Connection to DB client successful")

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=config.get("QDRANT_COLLECTION_NAME"),
        embedding=emb_model,
        vector_name=embedding_model,
    )

    logger.success("Vectorstore initialization successful")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # logger.info(f"Ma base de connaissance du site Insee comporte {len(db.get()["documents"])} documents")

    logger.info("------ rag_chain initialized, ready for use")

    return retriever


retriever = load_retriever_cache()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )


@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content="Recherche des documents les plus pertinents").send()

    best_documents = retriever.invoke(message.content)
    best_documents_df = [docs.metadata for docs in best_documents]
    best_documents_df = pd.DataFrame(best_documents_df)

    await cl.Message(content="Documents trouvés, je génère maintenant une réponse personnalisée").send()

    logger.debug(best_documents_df)

    context = format_docs(best_documents)
    # await cl.Message(content=context).send()

    question_with_context = prompt.format(question=message.content, context=context)

    message_history = cl.user_session.get("message_history")

    message_history.append({"role": "user", "content": question_with_context})

    msg = cl.Message(content="")

    stream = await vllm_client_completion.chat.completions.create(messages=message_history, stream=True, **settings)

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()

    elements = [cl.Dataframe(data=best_documents_df, display="inline", name="Dataframe")]
    await cl.Message(content="Documents les plus pertinents", elements=elements).send()
