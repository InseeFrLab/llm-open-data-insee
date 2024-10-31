import logging

import chainlit as cl
import s3fs
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.llms import VLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import load_config, simple_argparser
from src.db_building import load_retriever
from utils import create_prompt_from_instructions, format_docs, retrieve_db_from_cache

# Logging configuration
logger = logging.getLogger(__name__)

config = load_config(simple_argparser())["chainlit.app"]

fs = s3fs.S3FileSystem(endpoint_url=config["s3_endpoint_url"])

# PARAMETERS --------------------------------------

CLI_MESSAGE_SEPARATOR = (int(config.get("CLI_MESSAGE_SEPARATOR_LENGTH")) * "-") + " \n"

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

    logger.info(f"------ downloading {config["LLM_MODEL"]} or using from cache")

    # retrieve_llm_from_cache(model_id=LLM_MODEL)

    logger.info("------ database loaded")

    db = retrieve_db_from_cache(filesystem=fs, run_id=config["DATABASE_RUN_ID"])

    logger.info("------ database loaded")

    retriever, vectorstore = await cl.make_async(load_retriever)(
        emb_model_name=config["emb_model"],
        vectorstore=db,
        retriever_params={"search_type": "similarity", "search_kwargs": {"k": 10}},
    )

    db_docs = db.get()["documents"]
    ndocs = f"Ma base de connaissance du site Insee comporte {len(db_docs)} documents"
    logger.info(ndocs)

    logger.info("------ retriever ready for use")

    llm = VLLM(
        model=config["LLM_MODEL"],
        max_new_tokens=config["MAX_NEW_TOKEN"],
        top_p=config["TOP_P"],
        temperature=config["TEMPERATURE"],
        rep_penalty=config["REP_PENALTY"],
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

    async for _ in rag_chain.astream(
        message.content, config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)])
    ):
        await answer_msg.send()
        await cl.sleep(1)

    # TODO: ajouter un callback
    # https://docs.chainlit.io/api-reference/integrations/langchain#final-answer-streaming

    # await cl.Message(content=response).send()
