import os
from pathlib import Path

from langchain_core.prompts import PromptTemplate
import chainlit as cl

from config import RAG_PROMPT_TEMPLATE
from model_building import build_llm_model
from chain_building.build_chain import (
    load_retriever,
    build_chain
    )


PROJECT_PATH = Path(__file__).resolve().parents[1]

# S3 configuration
# S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
# fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

# Import Chroma DB from S3
# DB_PATH_S3 = os.path.join(os.environ["S3_BUCKET"], os.environ["DB_KEY_S3"])
DB_PATH_LOCAL = os.path.join(PROJECT_PATH, "data", "chroma_db")
# fs.get(DB_PATH_S3, DB_PATH_LOCAL, recursive=True)


@cl.on_chat_start
async def on_chat_start():

    # Generate prompt template
    prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)

    # Create a pipeline with tokenizer and LLM
    retriever = load_retriever(DB_PATH_LOCAL)
    llm = build_llm_model(quantization_config=True, config=True, token=os.environ["HF_TOKEN"])
    chain = build_chain(retriever, prompt, llm)

    # Declare runnable in chainlit
    cl.user_session.set("runnable", chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("runnable")
    response = await chain.arun(question=message.content,
                                callbacks=[cl.AsyncLangchainCallbackHandler()]
                                )

    await cl.Message(content=response["answer"]).send()
