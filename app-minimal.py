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
from src.db_building import load_retriever, load_vector_database
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


# PARAMETERS --------------------------------------

model = os.getenv("LLM_MODEL_NAME")
quantization = True
max_new_tokens = 10
model_temperature = 1
embedding = os.getenv("EMB_MODEL_NAME", EMB_MODEL_NAME)
CHROMA_DB_LOCAL_DIRECTORY = 'data/chroma_db'

os.environ['MLFLOW_TRACKING_URI'] = "https://projet-llm-insee-open-data-mlflow.user.lab.sspcloud.fr/"
fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": f"""https://{os.environ["AWS_S3_ENDPOINT"]}"""})

# APPLICATION -----------------------------------------


@cl.on_chat_start
async def on_chat_start():
    # Initial message
    init_msg = cl.Message(
        content="Bienvenue sur le ChatBot de l'INSEE!"
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
            await cl.Message(
                content="Vous avez choisi de partager vos interactions."
            ).send()
        if res and res.get("value") == "no_log":
            IS_LOGGING_ON = False
            await cl.Message(
                content="Vous avez choisi de garder vos interactions avec le ChatBot privées."
            ).send()
    cl.user_session.set("IS_LOGGING_ON", IS_LOGGING_ON)


    # Build chat model
    RETRIEVER_ONLY = str_to_bool(os.getenv("RETRIEVER_ONLY", "false"))
    cl.user_session.set("RETRIEVER_ONLY", RETRIEVER_ONLY)
    logging.info("------ chatbot mode : RAG")


    db = await cl.make_async(load_vector_database)(
        filesystem=fs,
        database_run_id="32d4150a14fa40d49b9512e1f3ff9e8c"
    )

    llm, tokenizer = await cl.make_async(build_llm_model)(
        model_name=model,
        quantization_config=quantization,
        config=True,
        token=os.getenv("HF_TOKEN"),
        streaming=False,
        generation_args={
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "do_sample": True,
            "temperature": model_temperature,
        },
    )
    logging.info("------llm loaded")

    # Set Validator chain in chainlit session
    validator = build_chain_validator(evaluator_llm=llm, tokenizer=tokenizer)
    cl.user_session.set("validator", validator)
    logging.info("------validator loaded")

    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        CHATBOT_TEMPLATE, tokenize=False, add_generation_prompt=True
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE
    )
    logging.info("------prompt loaded")
    retriever, vectorstore = await cl.make_async(load_retriever)(
                emb_model_name="OrdalieTech/Solon-embeddings-large-0.1",
                vectorstore=db, 
                persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
                retriever_params={"search_type": "similarity", "search_kwargs": {"k": 30}},
            )
    logging.info("------retriever loaded")
    logging.info(f"----- {len(vectorstore.get()['documents'])} documents")

    # Build chain
    RERANKING_METHOD = os.getenv("RERANKING_METHOD")
    if RERANKING_METHOD == "":
        RERANKING_METHOD = None
    chain = build_chain(
        retriever=retriever,
        prompt=prompt,
        llm=llm,
        bool_log=IS_LOGGING_ON,
        reranker=RERANKING_METHOD,
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
        answer_msg = cl.Message(content="")
        sources = list()
        titles = list()

        # Generate ChatBot's answer
        async for chunk in chain.astream(
            message.content,
            config=RunnableConfig(
                callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]
            ),
        ):
            if "answer" in chunk:
                await answer_msg.stream_token(chunk["answer"])
                generated_answer = chunk["answer"]

            if "context" in chunk:
                docs = chunk["context"]
                for doc in docs:
                    sources.append(doc.metadata.get("source"))
                    titles.append(doc.metadata.get("title"))

        await answer_msg.send()
        await cl.sleep(1)

        # Add sources to answer
        sources_msg = cl.Message(
            content=add_sources_to_messages(message="", sources=sources, titles=titles),
        )
        await sources_msg.send()

        # Log Q/A
        if cl.user_session.get("IS_LOGGING_ON"):
            embedding_model_name = os.getenv("EMB_MODEL_NAME")
            LLM_name = os.getenv("LLM_MODEL_NAME")
            reranker = os.getenv("RERANKING_METHOD")

            log_qa_to_s3(
                thread_id=message.thread_id,
                message_id=sources_msg.id,
                user_query=message.content,
                generated_answer=(
                    None if cl.user_session.get("RETRIEVER_ONLY") else generated_answer
                ),
                retrieved_documents=docs,
                embedding_model_name=embedding_model_name,
                LLM_name=None if cl.user_session.get("RETRIEVER_ONLY") else LLM_name,
                reranker=reranker,
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
            feedback_comment=feedback.comment,
        )


# Enable data persistence for human feedbacks
cl_data._data_layer = CustomDataLayer()