from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


def create_vectorstore(
    emb_model_name: str,
    persist_directory: str = "data/chroma_db",
    device: str = "cuda",
    collection_name: str = "insee_data",
):
    # Load Embedding model
    embeddings = HuggingFaceEmbeddings(model_name=emb_model_name, model_kwargs={"device": device})
    # Load vector database
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


def load_retriever(
    emb_model_name,
    persist_directory="data/chroma_db",
    device="cuda",
    collection_name: str = "insee_data",
):
    # Load vector database
    vectorstore = create_vectorstore(
        emb_model_name=emb_model_name, persist_directory=persist_directory, device=device
    )

    # Set up a retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"score_threshold": 0.5, "k": 10}
    )
    return retriever


def format_docs(docs) -> str:
    """
    Format the retrieved document before giving their content to complete the prompt
    """
    return "\n\n".join(docs)


def build_chain(retriever, prompt, llm):
    """
    Build a LLM chain based on Langchain package and INSEE data
    """
    # Create a Langchain LLM Chain
    chain = (
        {"context": format_docs | retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": format_docs | retriever, "question": RunnablePassthrough()}
    ).assign(answer=chain)

    return rag_chain_with_source
