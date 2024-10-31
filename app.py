import logging
import os

import chainlit as cl
import s3fs
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.llms import VLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
)

from src.config import load_config, simple_argparser
from src.db_building import load_retriever
from utils import create_prompt_from_instructions, format_docs, retrieve_db_from_cache

# Logging configuration
logger = logging.getLogger(__name__)

config = load_config(simple_argparser())["chainlit.app"]

fs = s3fs.S3FileSystem(endpoint_url=config["s3_endpoint_url"])

# PARAMETERS --------------------------------------

config["UVICORN_TIMEOUT_KEEP_ALIVE"] = "0"

CLI_MESSAGE_SEPARATOR = (config["CLI_MESSAGE_SEPARATOR_LENGTH"] * "-") + " \n"
DEFAULT_MAX_NEW_TOKENS = 10
DEFAULT_MODEL_TEMPERATURE = 1
embedding = config.get("EMB_MODEL")

QUANTIZATION = config.get("QUANTIZATION")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS))
MODEL_TEMPERATURE = int(os.getenv("MODEL_TEMPERATURE", DEFAULT_MODEL_TEMPERATURE))
RETURN_FULL_TEXT = os.getenv("RETURN_FULL_TEXT", True)
DO_SAMPLE = os.getenv("DO_SAMPLE", True)

DATABASE_RUN_ID = "32d4150a14fa40d49b9512e1f3ff9e8c"
LLM_MODEL = config.get("LLM_MODEL")
MAX_NEW_TOKEN = 8192
TEMPERATURE = 0.2
REP_PENALTY = 1.1
TOP_P = 0.8


# PROMPT -------------------------------------

system_instructions = """
Tu es un assistant spécialisé dans la statistique publique. Tu réponds à des questions concernant les données de l'Insee, l'institut national statistique Français.

Réponds en FRANCAIS UNIQUEMENT. Utilise une mise en forme au format markdown.

En utilisant UNIQUEMENT les informations présentes dans le contexte, réponds de manière argumentée à la question posée.

La réponse doit être développée et citer ses sources (titre et url de la publication) qui sont référencées à la fin. Cite notamment l'url d'origine de la publication, dans un format markdown.

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


# CHAT START -------------------------------


def format_docs2(docs: list):
    return "\n\n".join(
        [
            f"""
            Doc {i + 1}:\nTitle: {doc.metadata.get("Header 1")}\n
            Source: {doc.metadata.get("url")}\n
            Content:\n{doc.page_content}
            """
            for i, doc in enumerate(docs)
        ]
    )


@cl.on_chat_start
async def on_chat_start():
    # Initial message
    init_msg = cl.Message(content="Bienvenue sur le ChatBot de l'INSEE!")
    await init_msg.send()

    logger.info(f"------ downloading {LLM_MODEL} or using from cache")

    # retrieve_llm_from_cache(model_id=LLM_MODEL)

    logger.info("------ database loaded")

    db = retrieve_db_from_cache(filesystem=fs, run_id=DATABASE_RUN_ID)

    logger.info("------ database loaded")

    retriever, vectorstore = await cl.make_async(load_retriever)(
        emb_model_name=embedding,
        vectorstore=db,
        retriever_params={"search_type": "similarity", "search_kwargs": {"k": 10}},
    )

    db_docs = db.get()["documents"]
    ndocs = f"Ma base de connaissance du site Insee comporte {len(db_docs)} documents"
    logger.info(ndocs)

    logger.info("------ retriever ready for use")

    llm = VLLM(
        model=LLM_MODEL,
        max_new_tokens=MAX_NEW_TOKEN,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        rep_penalty=REP_PENALTY,
    )

    logger.info("------ VLLM object ready")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )

    cl.user_session.set("rag_chain", rag_chain)

    logger.info("------ rag_chain initialized, ready for use")
    logger.info(f"Thread ID : {init_msg.thread_id}")


@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")

    answer_msg = cl.Message(content="")

    async for chunk in rag_chain.astream(
        message.content, config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)])
    ):
        await answer_msg.send()
        await cl.sleep(1)

    # TODO: ajouter un callback
    # https://docs.chainlit.io/api-reference/integrations/langchain#final-answer-streaming

    # await cl.Message(content=response).send()
