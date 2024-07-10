import logging
import os

import chainlit as cl
import chainlit.data as cl_data
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.prompts import PromptTemplate

from src.chain_building.build_chain import build_chain
from src.chain_building.build_chain_validator import build_chain_validator
from src.db_loading import load_retriever
from src.model_building import build_llm_model
from src.results_logging.log_conversations import log_feedback_to_s3, log_qa_to_s3
from src.utils.formatting_utilities import add_sources_to_messages, str_to_bool

# Logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%Y-%m-%d %I:%M:%S %p",
                    level=logging.DEBUG
                    )


# Chatbot configuration
CHATBOT_INSTRUCTION = """
En utilisant UNIQUEMENT les informations présentes dans le contexte, réponds de manière argumentée à la question posée.
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
    {"role": "user", "content": """Tu es un assistant spécialisé dans la statistique publique.
    Tu réponds à des questions concernant les données de l'Insee, l'institut national statistique Français.
    Réponds en FRANCAIS UNIQUEMENT."""},
    {"role": "assistant", "content": CHATBOT_INSTRUCTION},
    {"role": "user", "content": USER_INSTRUCTION},
]


@cl.on_chat_start
async def on_chat_start():
    # Initial message
    init_msg = cl.Message(content="Bienvenue sur le ChatBot de l'INSEE!", disable_feedback=True)
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

    # Set Validator chain in chainlit session
    llm, tokenizer = build_llm_model(
        model_name=os.getenv("LLM_MODEL_NAME"),
        quantization_config=True,
        config=True,
        token=os.getenv("HF_TOKEN"),
        streaming=False,
        generation_args={
            "max_new_tokens": 10,
            "return_full_text": False,
            "do_sample": False
        },
    )
    # Set Validator chain in chainlit session
    validator = build_chain_validator(evaluator_llm=llm, tokenizer=tokenizer)
    cl.user_session.set("validator", validator)
    logging.info("------validator loaded")

    # Build chat model
    RETRIEVER_ONLY = str_to_bool(os.getenv("RETRIEVER_ONLY", 'false'))
    cl.user_session.set("RETRIEVER_ONLY", RETRIEVER_ONLY)
    if RETRIEVER_ONLY:
        logging.info("------ chatbot mode : retriever only")
        llm = None
        prompt = None
        retriever = load_retriever(
            emb_model_name=os.getenv("EMB_MODEL_NAME"),
            persist_directory="./data/chroma_db",
            retriever_params={
                            "search_type": "similarity",
                            "search_kwargs": {"k": 30}
                            }
        )
        logging.info("------retriever loaded")
       
    else:
        logging.info("------ chatbot mode : RAG")

        llm, tokenizer = build_llm_model(
            model_name=os.getenv("LLM_MODEL_NAME"),
            quantization_config=True,
            config=True,
            token=os.getenv("HF_TOKEN"),
            streaming=False,
            generation_args={
                "max_new_tokens": 2000,
                "return_full_text": False,
                "do_sample": True,
                "temperature": 0.2,
            }
        )
        logging.info("------llm loaded")

        RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(CHATBOT_TEMPLATE,
                                                            tokenize=False,
                                                            add_generation_prompt=True
                                                            )
        prompt = PromptTemplate(input_variables=["context", "question"],
                                template=RAG_PROMPT_TEMPLATE)
        logging.info("------prompt loaded")
        retriever = load_retriever(
                emb_model_name=os.getenv("EMB_MODEL_NAME"),
                persist_directory="./data/chroma_db", 
                retriever_params={
                    "search_type": "similarity",
                    "search_kwargs": {"k": 30}
                    }
                )
        logging.info("------retriever loaded")

    # Build chain
    RERANKING_METHOD = os.getenv("RERANKING_METHOD", None)
    if RERANKING_METHOD == "":
        RERANKING_METHOD = None 
    chain = build_chain(retriever=retriever, 
                prompt=prompt, 
                llm=llm,
                bool_log=IS_LOGGING_ON,
                reranker=RERANKING_METHOD
            )
    cl.user_session.set("chain", chain)
    logging.info("------chain built")

    logging.info(f"Thread ID : {init_msg.thread_id}")


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages and process the response using the RAG chain.
    """
    validator = cl.user_session.get("validator")
    test_relevancy = await check_query_relevance(
        validator=validator, query=message.content
    )
    if test_relevancy:
        # Retrieve the chain from the user session
        chain = cl.user_session.get("chain")

        # Initialize ChatBot's answer
        answer_msg = cl.Message(content="", disable_feedback=True)
        sources = list()
        titles = list()

        # Generate ChatBot's answer
        async for chunk in chain.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)])
        ):

            if 'answer' in chunk:
                await answer_msg.stream_token(chunk["answer"])
                generated_answer = chunk["answer"]

            if "context" in chunk:
                docs = chunk["context"]
                for doc in docs:
                    sources.append(doc.metadata.get("source", None))
                    titles.append(doc.metadata.get("title", None))

        await answer_msg.send()
        await cl.sleep(1)

        # Add sources to answer
        sources_msg = cl.Message(content=add_sources_to_messages(message="",
                                                                sources=sources,
                                                                titles=titles
                                                                ),
                                disable_feedback=False)
        await sources_msg.send()

        # Log Q/A
        if cl.user_session.get("IS_LOGGING_ON"):
            embedding_model_name = os.getenv("EMB_MODEL_NAME")
            LLM_name = os.getenv("LLM_MODEL_NAME", None)
            reranker = os.getenv("RERANKING_METHOD", None)

            log_qa_to_s3(
                thread_id=message.thread_id,
                message_id=sources_msg.id,
                user_query=message.content,
                generated_answer=None if cl.user_session.get("RETRIEVER_ONLY") else generated_answer,
                retrieved_documents=docs,
                embedding_model_name=embedding_model_name,
                LLM_name=None if cl.user_session.get("RETRIEVER_ONLY") else LLM_name,
                reranker=reranker
            )
    else:
        await cl.Message(
            content=f"Votre requête '{message.content}' ne concerne pas les domaines d'expertise de l'INSEE."
        ).send()

async def check_query_relevance(validator, query):

    result = await validator.ainvoke(
        query, config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler()])
    )
    return result

class CustomDataLayer(cl_data.BaseDataLayer):
    async def upsert_feedback(self, feedback: cl_data.Feedback) -> str:
        log_feedback_to_s3(
            thread_id=feedback.threadId,
            message_id=feedback.forId,
            feedback_value=feedback.value,
            feedback_comment=feedback.comment
            )


# Enable data persistence for human feedbacks
cl_data._data_layer = CustomDataLayer()
