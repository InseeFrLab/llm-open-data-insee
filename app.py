import os

from langchain_core.prompts import PromptTemplate
import chainlit as cl

from src.config import RAG_PROMPT_TEMPLATE, EMB_MODEL_NAME, MODEL_NAME
from src.model_building import build_llm_model
from src.chain_building.build_chain import (
    load_retriever,
    build_chain
    )

from dotenv import load_dotenv

load_dotenv()

@cl.on_chat_start
async def on_chat_start():
    # Set up RAG chain
    prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)
    retriever = load_retriever(emb_model_name=EMB_MODEL_NAME,
                               persist_directory="data/chroma_db")
    llm = build_llm_model(model_name=MODEL_NAME,
                          quantization_config=True,
                          config=True,
                          token=os.environ["HF_TOKEN"])
    chain = build_chain(retriever, prompt, llm)

    # Set RAG chain in chainlit session
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve RAG chain
    chain = cl.user_session.get("chain")

    # Process user query
    inputs = {"question": message.content}
    result = await chain.ainvoke(inputs)
    msg = cl.Message(content=result["answer"], disable_feedback=True)
    await msg.send()


# @cl.on_message
# async def on_message(message: cl.Message):
#     runnable = cl.user_session.get("runnable")

#     msg = cl.Message(content="")

#     for chunk in await cl.make_async(runnable.stream)(
#         {"question": message.content},
#         config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler(
#             stream_final_answer=True
#         )]),
#     ):
#         await msg.stream_token(chunk)

#     await msg.send()
