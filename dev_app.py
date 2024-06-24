from src.config import EMB_MODEL_NAME, MODEL_NAME
from src.model_building import build_llm_model
from src.chain_building.build_chain import (
    load_retriever,
    build_chain
    )


import os
import sys 
import logging

sys.path.append(".")

from dotenv import load_dotenv

# API and UX functions
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from chainlit.input_widget import Select

load_dotenv()

CHATBOT_INSTRUCTION = """
Tu es un assistant spécialisé dans la statistique publique répondant aux questions d'agent de l'INSEE.
Réponds en FRANCAIS UNIQUEMENT.
Utilise UNIQUEMENT les informations présentes dans le contexte, réponds de manière argumentée à la question posée.
La réponse doit être développée et citer ses sources.

Si tu ne peux pas induire ta réponse du contexte, ne réponds pas.
"""
USER_INSTRUCTION = """Voici le contexte sur lequel tu dois baser ta réponse :
Contexte: 
{context}
---
Voici la question à laquelle tu dois répondre :
Question: {question}"""

CHATBOT_TEMPLATE = [
    {"role": "system", "content": CHATBOT_INSTRUCTION},
    {"role": "user", "content": USER_INSTRUCTION},
]

# Retrieve Hugging Face token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Hugging Face token not found in environment variables.")

@cl.on_chat_start
async def on_chat_start():
    # Set up RAG chain
    await cl.Message(content="Bienvenue sur la ChatBot de l'INSEE!").send()
    await cl.sleep(1)

    res = await cl.AskActionMessage(
        content="Autorisez-vous le partage de vos intéractions avec le ChatBot!",
        actions=[
            cl.Action(name="log", value="log", label="✅ Accepter"),
            cl.Action(name="no log", value="no_log", label="❌ Refuser"),
        ],
        ).send()

    bool_log = False
    if res and res.get("value") == "log":
        await cl.Message(content="Vous avez choisi de partager vos intéractions.").send()
        bool_log = True
    if res and res.get("value") == "no_log":
        bool_log = False
        await cl.Message(content="Vous avez choisi de garder vos intéractions avec le ChatBot privées.").send()


    llm, tokenizer = build_llm_model(
        model_name=MODEL_NAME,
        quantization_config=True,
        config=True,
        token=HF_TOKEN,
        streaming=False 
        )
    logging.info("------llm loaded")
    
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(chat=CHATBOT_TEMPLATE, 
                                                        tokenize=False,
                                                        add_generation_prompt=True
                                                        )

    # load chain components
    prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)
    logging.info("------prompt loaded")
    retriever = load_retriever(emb_model_name=EMB_MODEL_NAME, persist_directory="./data/chroma_db")
    logging.info("------retriever loaded")

    # Allow the user to select their preferred reranker model
    await cl.Message(content="Choisissez votre méthode de reranking en paramètres").send()
    await cl.sleep(5)
    reranker_setting = await cl.ChatSettings(
        [
            Select(
                id="Reranker",
                label="Reranker models",
                values=["Aucun", "BM25", "Cross-encoder", "ColBERT", "Ensemble"],
                initial_index=0,
                description="Choisissez votre modèle de reranker"
            )
        ]
    ).send()

    if reranker_setting:
        reranker = None if reranker_setting["Reranker"] == "Aucun" else reranker_setting["Reranker"]

    chain = build_chain(retriever, prompt, llm, bool_log=bool_log, reranker=reranker)
    logging.info("------chain built")

    # Set RAG chain in chainlit session
    cl.user_session.set("chain", chain)

def add_sources_to_messages(message: str, sources: list, titles: list, topk : int = 5):
    """
    Append a list of sources and titles to a Chainlit message.

    Args:
    - message (str): The Chainlit message content to which the sources and titles will be added.
    - sources (list): A list of sources to append to the message.
    - titles (list): A list of titles to append to the message.
    - topk (int) : number of displayed sources. 
    """
    if len(sources) == len(titles):
        formatted_sources = f"\n\nSources (Top {topk}):\n" + "\n".join([f"{i+1}. {title} ({source})" for i, (source, title) in enumerate(zip(sources, titles)) if i < topk])
        message += formatted_sources
    else:
        message += "\n\nNo Sources available"

    return message

@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages and process the response using the RAG chain.
    """
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")

    # Initialize variables
    msg = cl.Message(content="")
    sources = list()
    titles = list()

    async for chunk in chain.astream(
        message.content, 
        config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]),
        ):

        if 'answer' in chunk:
            await msg.stream_token(chunk["answer"])
        
        if "context" in chunk:
            docs = chunk["context"]
            for i, doc in enumerate(docs):
                meta = doc.metadata
                sources.append(meta.get("source", None)) 
                titles.append(meta.get("title", None)) 
        
    await msg.send()
    await cl.sleep(1)
    msg_sources = cl.Message(content=add_sources_to_messages(
                            message="", 
                            sources=sources, 
                            titles=titles), 
                            disable_feedback=False)
    await msg_sources.send()

"""
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Demandez une définition",
            message="Quelle est la définition du déficit public?",
            icon="/public/insee_logo.png",
            ),
        cl.Starter(
            label="Demandez un chiffre précis",
            message="Quelles sont les statistiques sur l'espérance de vie en France ?",
            icon="/public/insee_logo.png",
            ),
        cl.Starter(
            label="Connaitre une méthodologie",
            message="Comment l'INSEE collecte-t-elle les données sur l'emploi ?",
            icon="/public/insee_logo.png",
            ),
        ]
"""