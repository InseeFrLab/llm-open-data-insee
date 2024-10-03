import logging
import os
import s3fs

import chainlit as cl
import chainlit.data as cl_data
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.prompts import PromptTemplate

from src.chain_building.build_chain import build_chain
from src.chain_building.build_chain_validator import build_chain_validator
from src.config import CHATBOT_TEMPLATE, EMB_MODEL_NAME
from src.db_building import (
    load_retriever,
    load_vector_database
)
from src.model_building import build_llm_model
from src.results_logging.log_conversations import log_feedback_to_s3, log_qa_to_s3
from src.utils.formatting_utilities import add_sources_to_messages, str_to_bool

# Logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.DEBUG,
)

# Remote file configuration
os.environ['MLFLOW_TRACKING_URI'] = "https://projet-llm-insee-open-data-mlflow.user.lab.sspcloud.fr/"
fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": f"""https://{os.environ["AWS_S3_ENDPOINT"]}"""})

# PARAMETERS --------------------------------------

os.environ['UVICORN_TIMEOUT_KEEP_ALIVE'] = "0"

model = os.getenv("LLM_MODEL_NAME")
CHROMA_DB_LOCAL_DIRECTORY = "./data/chroma_db"
CLI_MESSAGE_SEPARATOR = f"{80*'-'} \n"
quantization = True
DEFAULT_MAX_NEW_TOKENS = 10
DEFAULT_MODEL_TEMPERATURE = 1
embedding = os.getenv("EMB_MODEL_NAME", EMB_MODEL_NAME)

LLM_MODEL = os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
QUANTIZATION = os.getenv("QUANTIZATION", True)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS))
MODEL_TEMPERATURE = int(os.getenv("MODEL_TEMPERATURE", DEFAULT_MODEL_TEMPERATURE))
RETURN_FULL_TEXT = os.getenv("RETURN_FULL_TEXT", True)
DO_SAMPLE = os.getenv("DO_SAMPLE", True)

DATABASE_RUN_ID = "32d4150a14fa40d49b9512e1f3ff9e8c"


def retrieve_model_tokenizer_and_db(
    filesystem=fs,
    with_db=True
    #**kwargs
):

    # ------------------------
    # I - LOAD VECTOR DATABASE

    # Load LLM in session
    llm, tokenizer = build_llm_model(
            model_name=LLM_MODEL,
            quantization_config=QUANTIZATION,
            config=True,
            token=os.getenv("HF_TOKEN"),
            streaming=False,
            generation_args={
                "max_new_tokens": MAX_NEW_TOKENS,
                "return_full_text": RETURN_FULL_TEXT,
                "do_sample": DO_SAMPLE,
                "temperature": MODEL_TEMPERATURE
            },
    )

    if with_db is False:
        return llm, tokenizer, None

    # Ensure production database is used
    db = load_vector_database(
        filesystem=fs,
        database_run_id="32d4150a14fa40d49b9512e1f3ff9e8c"
        # hard coded pour le moment
    )

    return llm, tokenizer, db


# APPLICATION -----------------------------------------


@cl.on_chat_start
async def on_chat_start():

    # Initial message
    init_msg = cl.Message(
        content="Bienvenue sur le ChatBot de l'INSEE!"#, disable_feedback=True
    )
    await init_msg.send()

    logging.info("Retriever only mode !")

    llm, tokenizer = build_llm_model(
                model_name=LLM_MODEL,
                quantization_config=QUANTIZATION,
                config=True,
                token=os.getenv("HF_TOKEN"),
                streaming=False,
                generation_args={
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "return_full_text": RETURN_FULL_TEXT,
                    "do_sample": DO_SAMPLE,
                    "temperature": MODEL_TEMPERATURE
                },
    )

    db = load_vector_database(
            filesystem=fs,
            database_run_id=DATABASE_RUN_ID
            # hard coded pour le moment
    )

    retriever, vectorstore = await cl.make_async(load_retriever)(
                emb_model_name=embedding,
                persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
                vectorstore=db,
                retriever_params={
                    "search_type": "similarity",
                    "search_kwargs": {"k": 30}
                },
            )
    logging.info("Retriever loaded !")

    db_docs = db.get()["documents"]
    ndocs = f"Ma base de connaissance du site Insee comporte {len(db_docs)} documents"
    logging.info(ndocs)

    logging.info("Retriever loaded !")


    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
                CHATBOT_TEMPLATE, tokenize=False, add_generation_prompt=True
            )
    prompt = PromptTemplate(
            input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE
    )    


    chain = build_chain(
            retriever=retriever,
            prompt=prompt,
            llm=llm,
            reranker="BM25",
    )
    
    cl.user_session.set("chain", chain)
    logging.info("------chain built")

    logging.info(f"Thread ID : {init_msg.thread_id}")


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages and process the response using the RAG chain.
    """
    chain = cl.user_session.get("chain")

    # Initialize ChatBot's answer
    answer_msg = cl.Message(content="")#, disable_feedback=True)
    sources = list()
    titles = list()

    # Generate ChatBot's answer
    async for chunk in chain.astream_log(
        message.content,
        config=RunnableConfig(
            callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]
        ),
    ):
        if "answer" in chunk:
            await answer_msg.stream_token(chunk["answer"])
            generated_answer = chunk["answer"]
            logging.info(f"------answer: {generated_answer}")

        if "context" in chunk:
            docs = chunk["context"]
            for doc in docs:
                sources.append(doc.metadata.get("url"))
                titles.append(doc.metadata.get("Header 1"))

    await answer_msg.send()
    await cl.sleep(1)

    # Add sources to answer
    sources_msg = cl.Message(
        content=add_sources_to_messages(message="", sources=sources, titles=titles)#,
        #disable_feedback=False,
    )
    await sources_msg.send()

