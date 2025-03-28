import logging

import chainlit as cl
import s3fs
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.llms import VLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import DefaultFullConfig, process_args
from src.db_building import load_retriever, load_vector_database
from src.model_building import cache_model_from_hf_hub
from src.utils import create_prompt_from_instructions, format_docs

# Configuration
process_args()  # Strict minimal arguments processing
config = DefaultFullConfig()
fs = s3fs.S3FileSystem(endpoint_url=config.s3_endpoint_url)
logger = logging.getLogger(__name__)

# PARAMETERS --------------------------------------

CLI_MESSAGE_SEPARATOR = (config.cli_message_separator_length * "-") + " \n"

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


@cl.on_chat_start
async def on_chat_start():
    # Initial message
    init_msg = cl.Message(content="Bienvenue sur le ChatBot de l'INSEE!")
    await init_msg.send()

    logger.info(f"------ downloading model `{config.llm_model}` or using from cache")
    cache_model_from_hf_hub(model_id=config.llm_model)
    logger.info("------ model loaded")

    logger.info("------ loading database")
    db = load_vector_database(filesystem=fs)
    logger.info("------ database loaded")

    retriever, vectorstore = await cl.make_async(load_retriever)(
        vectorstore=db,
        retriever_params={"search_type": "similarity", "search_kwargs": {"k": 10}},
    )

    logger.info(f"Ma base de connaissance du site Insee comporte {len(db.get()['documents'])} documents")

    logger.info("------ retriever ready for use")

    llm = VLLM(
        model=config.llm_model,
        max_new_tokens=config.max_new_token,
        top_p=config.top_p,
        temperature=config.temperature,
        rep_penalty=config.rep_penalty,
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
