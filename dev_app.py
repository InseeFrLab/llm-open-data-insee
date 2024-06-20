import os

from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
import sys 
import json

sys.path.append(".")

from src.config import RAG_PROMPT_TEMPLATE, EMB_MODEL_NAME, MODEL_NAME
from src.model_building import build_llm_model
from src.chain_building.build_chain import (
    load_retriever,
    build_chain
    )

from dotenv import load_dotenv

load_dotenv()

# Retrieve Hugging Face token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Hugging Face token not found in environment variables.")

@cl.on_chat_start
async def on_chat_start():
    # Set up RAG chain

    try:
        prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)
        retriever = load_retriever(emb_model_name=EMB_MODEL_NAME, persist_directory="./data/chroma_db")
        llm = build_llm_model(
            model_name=MODEL_NAME,
            quantization_config=True,
            config=True,
            token=HF_TOKEN,
            streaming=False 
            )
        chain = build_chain(retriever, prompt, llm)

        # Set RAG chain in chainlit session
        cl.user_session.set("chain", chain)
    except Exception as e:
        print(f"Error during chat start: {e}")

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

def add_sources_to_messages(message: str, sources: list, titles: list):
    """
    Append a list of sources and titles to a Chainlit message.

    Args:
    - message (cl.Message): The Chainlit message object to which the sources and titles will be added.
    - sources (list): A list of sources to append to the message.
    - titles (list): A list of titles to append to the message.
    """
    if len(sources) == len(titles):
        formatted_sources = "\n\nSources:\n" + "\n".join([f"{i+1}. {title} ({source})" for i, (source, title) in enumerate(zip(sources, titles))])
        message += formatted_sources
    else:
        message += "\n\nNo Sources available"

    return message

@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages and process the response using the RAG chain.
    """
    try:
        chain = cl.user_session.get("chain")
        if not chain:
            raise ValueError("Chain not found in user session.")

        # Initialize variables
        msg = cl.Message(content="")
        sources = list()
        titles = list()

        async for chunk in chain.astream(
            message.content, 
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
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
        await msg_sources.send()

    except Exception as e:
        error_msg = cl.Message(content=f"An error occurred: {e}", disable_feedback=True)
        await error_msg.send()


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

