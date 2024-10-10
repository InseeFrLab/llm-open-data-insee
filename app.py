from transformers import AutoTokenizer
import os
import s3fs
import logging

from src.db_building import load_retriever, load_vector_database

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import VLLM
import chainlit as cl


from src.config import CHATBOT_TEMPLATE, EMB_MODEL_NAME
from utils import (
    format_docs, create_prompt_from_instructions,
    retrieve_llm_from_cache,
    retrieve_db_from_cache
)

# Logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.DEBUG,
)

# Remote file configuration
os.environ["MLFLOW_TRACKING_URI"] = (
    "https://projet-llm-insee-open-data-mlflow.user.lab.sspcloud.fr/"
)
fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": f"""https://{os.environ["AWS_S3_ENDPOINT"]}"""}
)

# PARAMETERS --------------------------------------

os.environ["UVICORN_TIMEOUT_KEEP_ALIVE"] = "0"

CLI_MESSAGE_SEPARATOR = f"{80*'-'} \n"
quantization = True
DEFAULT_MAX_NEW_TOKENS = 10
DEFAULT_MODEL_TEMPERATURE = 1
embedding = os.getenv("EMB_MODEL_NAME", EMB_MODEL_NAME)

QUANTIZATION = os.getenv("QUANTIZATION", True)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS))
MODEL_TEMPERATURE = int(os.getenv("MODEL_TEMPERATURE", DEFAULT_MODEL_TEMPERATURE))
RETURN_FULL_TEXT = os.getenv("RETURN_FULL_TEXT", True)
DO_SAMPLE = os.getenv("DO_SAMPLE", True)

DATABASE_RUN_ID = "32d4150a14fa40d49b9512e1f3ff9e8c"
LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MAX_NEW_TOKEN = 8192
TEMPERATURE = 0.2
REP_PENALTY = 1.1
TOP_P = 0.8


# PROMPT -------------------------------------

system_instructions = """
Tu es un assistant spécialisé dans la statistique publique. Tu réponds à des questions concernant les données de l'Insee, l'institut national statistique Français.

Réponds en FRANCAIS UNIQUEMENT. Utilise une mise en forme au format markdown.

En utilisant UNIQUEMENT les informations présentes dans le contexte, réponds de manière argumentée à la question posée.

La réponse doit être développée et citer ses sources (titre et url de la publication) qui sont référencées à la fin. Cite notamment l'url d'origine de la publication, dans un format markdown.

Cite 5 sources maximum.

Tu n'es pas obligé d'utiliser les sources les moins pertinentes. 

Si tu ne peux pas induire ta réponse du contexte, ne réponds pas.

Voici le contexte sur lequel tu dois baser ta réponse :
Contexte: {context}
"""

question_instructions = """
Voici la question à laquelle tu dois répondre :
Question: {question}

Réponse:
"""


prompt = create_prompt_from_instructions(
    system_instructions, question_instructions
)


# CHAT START -------------------------------


@cl.on_chat_start
async def on_chat_start():

    # Initial message
    init_msg = cl.Message(content="Bienvenue sur le ChatBot de l'INSEE!")
    await init_msg.send()

    logging.info(f"------ downloading {LLM_MODEL} or using from cache")

    retrieve_llm_from_cache(model_id=LLM_MODEL)

    logging.info("------ database loaded")

    db = retrieve_db_from_cache(
        filesystem=fs,
        run_id=DATABASE_RUN_ID
    )

    logging.info("------ database loaded")

    retriever, vectorstore = load_retriever(
        emb_model_name=embedding,
        vectorstore=db,
        retriever_params={"search_type": "similarity", "search_kwargs": {"k": 10}},
    )

    logging.info("------ retriever ready for use")

    llm = VLLM(
        model=LLM_MODEL,
        max_new_tokens=MAX_NEW_TOKEN,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        rep_penalty=REP_PENALTY,
    )

    logging.info("------ VLLM object ready")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    cl.user_session.set("rag_chain", rag_chain)

    logging.info("------ rag_chain initialized, ready for use")
    logging.info(f"Thread ID : {init_msg.thread_id}")


@cl.on_message
async def on_message(message: cl.Message):
    
    #rag_chain = cl.user_session.get("rag_chain")
    #response = rag_chain.invoke(message.content)
    response = f"Hello, you just sent: {message.content}!"

    await cl.Message(content=response).send()
