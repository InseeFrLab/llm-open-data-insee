import os

from langchain_core.prompts import PromptTemplate
import chainlit as cl

from src.config import RAG_PROMPT_TEMPLATE
from src.model_building import build_llm_model
from src.chain_building.build_chain import (
    load_retriever,
    build_chain
    )


@cl.on_chat_start
async def on_chat_start():

    # Generate prompt template
    prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)

    # Create a pipeline with tokenizer and LLM
    retriever = load_retriever(emb_model_name="sentence-transformers/all-MiniLM-L6-v2",
                               persist_directory="data/chroma_db")
    llm = build_llm_model(model_name="mistralai/Mistral-7B-Instruct-v0.2",
                          quantization_config=True,
                          config=True,
                          token=os.environ["HF_TOKEN"])
    chain = build_chain(retriever, prompt, llm)

    # Declare runnable in chainlit
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
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
