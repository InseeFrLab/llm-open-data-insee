import logging
import os
from datetime import datetime

import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.prompts import PromptTemplate

from src.chain_building.build_chain import build_chain, load_retriever
from src.model_building import build_llm_model
from src.results_logging.log_conv import log_conversation_to_s3
from src.utils.formatting_utilities import add_sources_to_messages, str_to_bool

# Logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%Y-%m-%d %I:%M:%S %p",
                    level=logging.DEBUG
                    )


CHATBOT_INSTRUCTION = """
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
    {"role": "user", "content": """Tu es un assistant spécialisé dans la statistique publique répondant aux questions d'agent de l'INSEE.
    Réponds en FRANCAIS UNIQUEMENT."""},
    {"role": "assistant", "content": CHATBOT_INSTRUCTION},
    {"role": "user", "content": USER_INSTRUCTION},
]


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Bienvenue sur le ChatBot de l'INSEE!").send()

    # Define conversation ID
    session_start_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = cl.user_session.get("id")
    conv_id = f'{session_start_timestamp}_{session_id}'
    cl.user_session.set("conv_id", conv_id)

    # Logging configuration
    IS_LOGGING_ON = True
    ASK_USER_BEFORE_LOGGING = str_to_bool(os.getenv("ASK_USER_BEFORE_LOGGING", "false"))
    if ASK_USER_BEFORE_LOGGING:
        res = await cl.AskActionMessage(
            content="Autorisez-vous le partage de vos intéractions avec le ChatBot!",
            actions=[
                cl.Action(name="log", value="log", label="✅ Accepter"),
                cl.Action(name="no log", value="no_log", label="❌ Refuser"),
            ],
            ).send()
        if res and res.get("value") == "log":
            await cl.Message(content="Vous avez choisi de partager vos intéractions.").send()
        if res and res.get("value") == "no_log":
            IS_LOGGING_ON = False
            await cl.Message(content="Vous avez choisi de garder vos intéractions avec le ChatBot privées.").send()
    cl.user_session.set("IS_LOGGING_ON", IS_LOGGING_ON)

    # Build chat model
    RETRIEVER_ONLY = str_to_bool(os.getenv("RETRIEVER_ONLY", 'false'))
    if RETRIEVER_ONLY:
        logging.info("------ chatbot mode : retriever only")
        llm = None
        prompt = None
        retriever = load_retriever(emb_model_name=os.getenv("EMB_MODEL_NAME"),
                                   persist_directory="./data/chroma_db")
        logging.info("------retriever loaded")
    else:
        logging.info("------ chatbot mode : RAG")

        llm, tokenizer = build_llm_model(
            model_name=os.getenv("LLM_MODEL_NAME"),
            quantization_config=True,
            config=True,
            token=os.getenv("HF_TOKEN"),
            streaming=False
            )
        logging.info("------llm loaded")

        RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(CHATBOT_TEMPLATE,
                                                            tokenize=False,
                                                            add_generation_prompt=True
                                                            )
        prompt = PromptTemplate(input_variables=["context", "question"],
                                template=RAG_PROMPT_TEMPLATE)
        logging.info("------prompt loaded")
        retriever = load_retriever(emb_model_name=os.getenv("EMB_MODEL_NAME"),
                                   persist_directory="./data/chroma_db")
        logging.info("------retriever loaded")

    # Build chain
    RERANKING_METHOD = os.getenv("RERANKING_METHOD", None)
    chain = build_chain(retriever, prompt, llm,
                        bool_log=IS_LOGGING_ON,
                        reranker=RERANKING_METHOD)
    logging.info("------chain built")

    # Set RAG chain in chainlit session
    cl.user_session.set("chain", chain)


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
        config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)])
    ):

        if 'answer' in chunk:
            await msg.stream_token(chunk["answer"])
            generated_answer = chunk["answer"]

        if "context" in chunk:
            docs = chunk["context"]
            for doc in docs:
                sources.append(doc.metadata.get("source", None))
                titles.append(doc.metadata.get("title", None))

    await msg.send()
    await cl.sleep(1)
    msg_sources = cl.Message(content=add_sources_to_messages(message="",
                                                             sources=sources,
                                                             titles=titles
                                                             ),
                             disable_feedback=False)
    await msg_sources.send()

    # Log Q/A
    if cl.user_session.get("IS_LOGGING_ON"):
        embedding_model_name = os.getenv("EMB_MODEL_NAME")
        LLM_name = os.getenv("LLM_MODEL_NAME")
        reranker = os.getenv("RERANKING_METHOD", None)

        log_conversation_to_s3(
            conv_id=cl.user_session.get("conv_id"),
            dir_s3=os.path.join(os.getenv("S3_BUCKET"), "data", "chatbot_logs"),
            user_query=message.content,
            retrieved_documents=docs,
            generated_answer=generated_answer,
            embedding_model_name=embedding_model_name,
            LLM_name=LLM_name,
            reranker=reranker
        )
