from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser

from config import EMB_MODEL_NAME, EMB_DEVICE
from config import DB_DIR
from config import RAG_PROMPT_TEMPLATE

from model_building import build_llm_model
from results_logging import log_chain_results
from utils import format_docs


def build_chain_with_logging():
    """
    Runs the chain and logs the results to a JSON file
    """
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMB_MODEL_NAME, model_kwargs={"device": EMB_DEVICE}
    )
    vectorstore = Chroma(
        collection_name="insee_data",
        embedding_function=hf_embeddings,
        persist_directory=str(DB_DIR),
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"score_threshold": 0.5, "k": 10}
    )
    prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)

    llm = build_llm_model()

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )
    # Create a chain that returns sources
    # and stores them into a log file
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs) | RunnableLambda(log_chain_results).bind(prompt=prompt)

    return rag_chain_with_source
