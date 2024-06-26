from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


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
    vectorstore = create_vectorstore(emb_model_name=emb_model_name, persist_directory=persist_directory, device=device)

    # Set up a retriever
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"score_threshold": 0.5, "k": 10})
    return retriever
