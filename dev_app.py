from src.config import EMB_MODEL_NAME, MODEL_NAME
from src.model_building import build_llm_model
from src.chain_building.build_chain import (
    load_retriever,
    build_chain
    )
from src.chain_building.build_chain_with_logging import (
    build_chain_with_logging
)

import os
from langchain_core.prompts import PromptTemplate
# from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
import sys 
import logging

sys.path.append(".")

from dotenv import load_dotenv

load_dotenv()

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
    else:
        await cl.Message(content="Vous avez choisi de garder vos intéractions avec le ChatBot privées.").send()

    # load chain components
    prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)
    logging.info("prompt loaded")
    retriever = load_retriever(emb_model_name=EMB_MODEL_NAME, persist_directory="./data/chroma_db")
    logging.info("retriever loaded")
    llm = build_llm_model(
        model_name=MODEL_NAME,
        quantization_config=True,
        config=True,
        token=HF_TOKEN,
        streaming=False 
        )
    logging.info("llm loaded")

    if bool_log:
        chain = build_chain(retriever, prompt, llm)
    else:
        chain = build_chain_with_logging(retriever, prompt, llm)

    # Set RAG chain in chainlit session
    cl.user_session.set("chain", chain)

"""
@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve RAG chain
    chain = cl.user_session.get("chain")

    # Process user query
    inputs = {"question": message.content}
    result = await chain.ainvoke(inputs)
    msg = cl.Message(content=result["answer"], disable_feedback=True)
    await msg.send()
"""

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
    #initilize Asynchronous Callback Handler 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Answer"]
    )
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")

    res = await chain.acall(message.content, callbacks=[cb])

    if not cb.answer_reached:
        await cl.Message(content=res["text"]).send()

    """  # Initialize variables
    msg = cl.Message(content="")
    sources = list()
    titles = list()

    async for chunk in chain.astream(
        message.content, 
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True)]),
        ):

        if 'answer' in chunk:
            await msg.stream_token(chunk["answer"])
        
        # print("chunck keys : ", chunk.keys())
        if "context" in chunk:
            docs = chunk["context"]
            #print("number of retrieved documents : ", len(docs))
            for i, doc in enumerate(docs):
                #print(f"\nDoc {i} : {doc.metadata}")
                meta = doc.metadata # json.loads(doc.metadata) 
                sources.append(meta.get("source",None)) 
                titles.append(meta.get("title",None)) 
        
    #await msg.send()

    msg_sources = cl.Message(
        content=add_sources_to_messages(
                                message="", 
                                sources=sources, 
                                titles=titles),
                                disable_feedback=True)
    await msg_sources.send()"""
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