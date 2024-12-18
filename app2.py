import logging
import os
from typing import Any

import chainlit as cl
import s3fs
from langchain.schema.runnable.config import RunnableConfig
from langchain_chroma.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

from src.chain_building.build_chain import build_chain
from src.chain_building.build_chain_validator import build_chain_validator
from src.config import DefaultFullConfig, process_args
from src.db_building import load_retriever, load_vector_database
from src.model_building import build_llm_model
from src.results_logging.log_conversations import log_qa_to_s3
from src.utils.formatting_utilities import add_sources_to_messages, get_chatbot_template, str_to_bool

# Logging, configuration and S3
args = process_args()
config = DefaultFullConfig()
logger = logging.getLogger(__name__)
fs = s3fs.S3FileSystem(endpoint_url=config.s3_endpoint_url)

# PARAMETERS --------------------------------------

CLI_MESSAGE_SEPARATOR = (config.cli_message_separator_length * "-") + " \n"

# APPLICATION -----------------------------------------


def retrieve_model_tokenizer_and_db(filesystem=fs, with_db=True) -> tuple[HuggingFacePipeline, Any, Chroma | None]:
    # ------------------------
    # I - LOAD VECTOR DATABASE

    # Load LLM in session
    llm, tokenizer = build_llm_model(
        model_name=config.llm_model,
        streaming=False,
    )
    return (
        llm,
        tokenizer,
        (load_vector_database(filesystem=fs) if with_db else None),
    )


@cl.on_chat_start
async def on_chat_start():
    # Initial message
    init_msg = cl.Message(
        content="Bienvenue sur le ChatBot de l'INSEE!"  # , disable_feedback=True
    )
    await init_msg.send()

    # Logging configuration
    IS_LOGGING_ON = True
    ASK_USER_BEFORE_LOGGING = str_to_bool(os.getenv("ASK_USER_BEFORE_LOGGING", "false"))
    if ASK_USER_BEFORE_LOGGING:
        res = await cl.AskActionMessage(
            content="Autorisez-vous le partage de vos interactions avec le ChatBot!",
            actions=[
                cl.Action(name="log", value="log", label="✅ Accepter"),
                cl.Action(name="no log", value="no_log", label="❌ Refuser"),
            ],
        ).send()
        if res and res.get("value") == "log":
            await cl.Message(content="Vous avez choisi de partager vos interactions.").send()
        if res and res.get("value") == "no_log":
            IS_LOGGING_ON = False
            await cl.Message(content="Vous avez choisi de garder vos interactions avec le ChatBot privées.").send()
    cl.user_session.set("IS_LOGGING_ON", IS_LOGGING_ON)

    # -------------------------------------------------
    # I - CREATING RETRIEVER AND IMPORTING DATABASE

    # Build chat model
    RETRIEVER_ONLY = DefaultFullConfig().retriever_only
    cl.user_session.set("RETRIEVER_ONLY", RETRIEVER_ONLY)

    # Log on CLI to follow the configuration
    if RETRIEVER_ONLY:
        logger.info(f"{CLI_MESSAGE_SEPARATOR} \nchatbot mode : retriever only \n")
    else:
        logger.info(f"{CLI_MESSAGE_SEPARATOR} \nchatbot mode : RAG \n")

    llm = None
    prompt = None
    db = load_vector_database(filesystem=fs)
    retriever, vectorstore = await cl.make_async(load_retriever)(
        vectorstore=db,
        retriever_params={"search_type": "similarity", "search_kwargs": {"k": 30}},
    )
    logger.info("Retriever loaded !")

    if not RETRIEVER_ONLY:
        llm, tokenizer, db = await cl.make_async(retrieve_model_tokenizer_and_db)(
            filesystem=fs,
            with_db=True,
        )
        db_docs = db.get()["documents"]
        logger.info(f"Ma base de connaissance du site Insee comporte {len(db_docs)} documents")

        RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
            get_chatbot_template(), tokenize=False, add_generation_prompt=True
        )
        prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)
        logger.info("------prompt loaded")

    # Set Validator chain in chainlit session
    validator = build_chain_validator(evaluator_llm=llm, tokenizer=tokenizer)
    cl.user_session.set("validator", validator)
    logger.info("------validator loaded")

    # Build chain
    chain = build_chain(
        retriever=retriever,
        prompt=prompt,
        llm=llm,
        reranker=os.getenv("RERANKING_METHOD") or None,
    )
    cl.user_session.set("chain", chain)
    logger.info("------chain built")

    logger.info(f"Thread ID : {init_msg.thread_id}")


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages and process the response using the RAG chain.
    """
    # validator = cl.user_session.get("validator")
    # test_relevancy = await check_query_relevance(
    #    validator=validator, query=message.content
    # )
    test_relevancy = True
    if test_relevancy:
        # Retrieve the chain from the user session
        chain = cl.user_session.get("chain")

        # Initialize ChatBot's answer
        answer_msg = cl.Message(content="")  # , disable_feedback=True)
        sources = list()
        titles = list()

        # Generate ChatBot's answer
        async for chunk in chain.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]),
        ):
            if "answer" in chunk:
                await answer_msg.stream_token(chunk["answer"])
                generated_answer = chunk["answer"]

            if "context" in chunk:
                docs = chunk["context"]
                for doc in docs:
                    sources.append(doc.metadata.get("url"))
                    titles.append(doc.metadata.get("Header 1"))

        await answer_msg.send()
        await cl.sleep(1)

        # Add sources to answer
        sources_msg = cl.Message(
            content=add_sources_to_messages(message="", sources=sources, titles=titles)  # ,
            # disable_feedback=False,
        )
        await sources_msg.send()

        # Log Q/A
        if cl.user_session.get("IS_LOGGING_ON"):
            log_qa_to_s3(
                filesystem=fs,
                thread_id=message.thread_id,
                message_id=sources_msg.id,
                user_query=message.content,
                generated_answer=(None if cl.user_session.get("RETRIEVER_ONLY") else generated_answer),
                retrieved_documents=docs,
                llm_name=None if cl.user_session.get("RETRIEVER_ONLY") else config.llm_model,
            )
    else:
        await cl.Message(
            content=f"Votre requête '{message.content}' ne concerne pas les domaines d'expertise de l'INSEE."
        ).send()


# async def check_query_relevance(validator, query):
#     result = await validator.ainvoke(
#         query, config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler()])
#     )
#     return result


# class CustomDataLayer(cl_data.BaseDataLayer):
#     async def upsert_feedback(self, feedback: cl_data.Feedback) -> str:
#         log_feedback_to_s3(
#             thread_id=feedback.threadId,
#             message_id=feedback.forId,
#             feedback_value=feedback.value,
#             feedback_comment=feedback.comment,
#         )


# # Enable data persistence for human feedbacks
# cl_data._data_layer = CustomDataLayer()
