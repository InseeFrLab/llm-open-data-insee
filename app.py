import os
import logging

from langchain_core.prompts import PromptTemplate
import chainlit as cl

from src.model_building import build_llm_model
from src.chain_building.build_chain import (
    load_retriever,
    build_chain
    )


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%Y-%m-%d %I:%M:%S %p",
                    level=logging.DEBUG
                    )


RAG_PROMPT_TEMPLATE = """
<s>[INST]
Tu es un assistant spécialisé dans la statistique publique répondant aux questions d'agent de l'INSEE.
Réponds en Français seulement.
Utilise les informations obtenues dans le contexte, réponds de manière argumentée à la question posée.
La réponse doit être développée et citer ses sources.

Si tu ne peux pas induire ta réponse du contexte, ne réponds pas.
Voici le contexte sur lequel tu dois baser ta réponse :
Contexte: {context}
        ---
Voici la question à laquelle tu dois répondre :
Question: {question}
[/INST]
"""


@cl.on_chat_start
async def on_chat_start():
    # Set up RAG chain
    prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)

    retriever = load_retriever(emb_model_name=os.environ["EMB_MODEL_NAME"],
                               persist_directory="./data/chroma_db")
    logger.info("Retriever loaded.")
    llm, _  = build_llm_model(model_name=os.environ["LLM_MODEL_NAME"],
                          quantization_config=True,
                          config=True,
                          token=os.environ["HF_TOKEN"])
    logger.info("LLM loaded.")
    chain = build_chain(retriever, prompt, llm)

    # Set RAG chain in chainlit session
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve RAG chain
    chain = cl.user_session.get("chain")

    # Process user query
    inputs = message.content
    result = await chain.ainvoke(inputs)
    msg = cl.Message(content=result["answer"], disable_feedback=True)
    await msg.send()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Definition",
            message="Quelle est la définition du déficit public?",
            icon="/public/insee_logo.png",
            ),
        cl.Starter(
            label="Chiffre",
            message="Quelles sont les statistiques sur l'espérance de vie en France ?",
            icon="/public/insee_logo.png",
            ),
        cl.Starter(
            label="Méthodologie",
            message="Comment l'INSEE collecte-t-elle les données sur l'emploi ?",
            icon="/public/insee_logo.png",
            ),
        ]
