from openai import AsyncOpenAI
import chainlit as cl

from src.utils import create_prompt_from_instructions, format_docs
from src.db_building.build_database import load_vector_database_from_local
from src.db_building import load_retriever

from loguru import logger


client = AsyncOpenAI(
    base_url="https://projet-llm-insee-open-data-vllm.user.lab.sspcloud.fr/v1/", api_key="EMPTY")
# Instrument the OpenAI client
cl.instrument_openai()

settings = {
    "model": 'mistralai/Mistral-Small-24B-Instruct-2501',
    "temperature": 0,
    # ... more settings
}


# PROMPT -------------------------------------

system_instructions = """
Tu es un assistant spécialisé dans la statistique publique.
Tu réponds à des questions concernant les données de l'Insee, l'institut national statistique Français.

Réponds en FRANCAIS UNIQUEMENT. Utilise une mise en forme au format markdown.

En utilisant UNIQUEMENT les informations présentes dans le contexte, réponds de manière argumentée à la question posée.

La réponse doit être développée et citer ses sources (titre et url de la publication) qui sont référencées à la fin.
Cite notamment l'url d'origine de la publication, dans un format markdown.

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


prompt = create_prompt_from_instructions(system_instructions, question_instructions)

@cl.cache
def load_retriever_cache():
    logger.info("Loading vector database")

    db = load_vector_database_from_local(
        persist_directory="data/chroma_db_checkpoint/", embedding_model="OrdalieTech/Solon-embeddings-large-0.1"
    )

    retriever, vectorstore = load_retriever(
        vectorstore=db,
        retriever_params={"search_type": "similarity", "search_kwargs": {"k": 10}},
    )

    logger.info(f"Ma base de connaissance du site Insee comporte {len(db.get()["documents"])} documents")

    logger.info("------ rag_chain initialized, ready for use")

    return retriever


retriever = load_retriever_cache()


@cl.on_chat_start
async def on_chat_start():

    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )


@cl.on_message
async def on_message(message: cl.Message):

    best_documents = retriever.invoke(message.content)

    context = format_docs(best_documents)
    #await cl.Message(content=context).send()

    question_with_context = prompt.format(question=message.content, context=context)

    message_history = cl.user_session.get("message_history")

    message_history.append({"role": "user", "content": question_with_context})

    msg = cl.Message(content="")

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()