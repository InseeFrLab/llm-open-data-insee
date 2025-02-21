import os

import chainlit as cl

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from src.utils import create_prompt_from_instructions, format_docs
from src.model_building import cache_model_from_hf_hub

from loguru import logger
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from src.model_building import cache_model_from_hf_hub
from src.utils import create_prompt_from_instructions, format_docs

# ENVIRONEMENT ------------------------------

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "OrdalieTech/Solon-embeddings-large-0.1")
URL_QDRANT = os.getenv("URL_QDRANT", None)
API_KEY_QDRANT = os.getenv("API_KEY_QDRANT", None)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "dirag_solon")
logger.debug(f"Using {EMBEDDING_MODEL} for database retrieval")
logger.debug(f"Setting {URL_QDRANT} as vector database endpoint")

URL_VLLM_CLIENT = os.getenv("URL_VLLM_CLIENT")
logger.debug(f"Setting {URL_VLLM_CLIENT} for database retrieval")


vllm_client = AsyncOpenAI(base_url=URL_VLLM_CLIENT, api_key="EMPTY")
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


@cl.cache
def load_retriever_cache():
    logger.info("Loading vector database")

    cache_model_from_hf_hub(EMBEDDING_MODEL, hf_token=os.environ.get("HF_TOKEN"))

    emb_model = HuggingFaceEmbeddings(  # load from sentence transformers
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
            show_progress=False,
        )

    client = QdrantClient(
        url=URL_QDRANT,
        api_key=API_KEY_QDRANT,
        port="443",
        https="true"
    )

    logger.success("Connection to DB client successful")

    vectorstore = QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=emb_model, vector_name=EMBEDDING_MODEL
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

    stream = await vllm_client.chat.completions.create(messages=message_history, stream=True, **settings)

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()

    elements = [cl.Dataframe(data=best_documents_df, display="inline", name="Dataframe")]
    await cl.Message(content="Documents les plus pertinents", elements=elements).send()
