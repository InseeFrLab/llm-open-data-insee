from collections.abc import Sequence
from typing import Any

# loading rerankers
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from ragatouille import RAGPretrainedModel

from src.results_logging import log_chain_results
from src.utils import format_docs


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


RERANKER_CROSS_ENCODER = "dangvantuan/CrossEncoder-camembert-large"
RERANKER_COLBERT = "antoinelouis/colbertv2-camembert-L4-mmarcoFR"


# Define the compression function
def compress_documents_lambda(documents: Sequence[Document], query: str, k: int = 5, **kwargs: dict[str, Any]) -> Sequence[Document]:
    """Compress retrieved documents given the query context."""

    # Initialize the retriever with the documents
    retriever = BM25Retriever.from_documents(documents, k=k, **kwargs)
    return retriever.get_relevant_documents(query)


def build_chain(retriever, prompt: str, llm=None, bool_log: bool = False, reranker=None):
    """
    Build a LLM chain based on Langchain package and INSEE data
    """
    # Create a Langchain LLM Chain
    chain = RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) | prompt | llm | StrOutputParser()
    # Define the retrieval reranker strategy
    if reranker is None:
        retrieval_agent = retriever
    elif reranker == "BM25":
        retrieval_agent = RunnableParallel({"documents": retriever, "query": RunnablePassthrough()}) | RunnableLambda(
            lambda r: compress_documents_lambda(documents=r["documents"], query=r["query"], k=10)
        )
    elif reranker == "Cross-encoder":
        model = HuggingFaceCrossEncoder(model_name=RERANKER_CROSS_ENCODER)
        compressor = CrossEncoderReranker(model=model, top_n=10)
        retrieval_agent = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    elif reranker == "ColBERT":
        colBERT = RAGPretrainedModel.from_pretrained(RERANKER_COLBERT)
        retrieval_agent = ContextualCompressionRetriever(base_compressor=colBERT.as_langchain_document_compressor(k=10), base_retriever=retriever)
    elif reranker == "Ensemble":
        # BM25
        reranker_1 = RunnableParallel({"documents": retriever, "query": RunnablePassthrough()}) | RunnableLambda(
            lambda r: compress_documents_lambda(documents=r["documents"], query=r["query"], k=10)
        )
        # Cross encoder
        compressor = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=RERANKER_CROSS_ENCODER), top_n=10)
        reranker_2 = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

        # ColBERT
        reranker_3 = ContextualCompressionRetriever(
            base_compressor=RAGPretrainedModel.from_pretrained(RERANKER_COLBERT).as_langchain_document_compressor(k=10),
            base_retriever=retriever,
        )

        retrieval_agent = EnsembleRetriever(retrievers=[reranker_1, reranker_2, reranker_3], weigths=[1 / 3, 1 / 3, 1 / 3])
    else:
        raise ValueError("This reranking method is not handled by the ChatBot or does not exist")

    # build the first part of the chain
    rag_chain_with_source = RunnableParallel({"context": retrieval_agent, "question": RunnablePassthrough()}).assign(answer=chain)

    if bool_log:
        return rag_chain_with_source | RunnableLambda(log_chain_results).bind(prompt=prompt, reranker=reranker)
    else:
        return rag_chain_with_source


def build_chain_retriever(retriever, bool_log: bool = False, reranker=None):
    """
    Build a langchain chain without generation, focusing on retrieving right ressources
    """

    # Define the retrieval reranker strategy
    if reranker is None:
        retrieval_agent = retriever
    elif reranker == "BM25":
        retrieval_agent = RunnableParallel({"documents": retriever, "query": RunnablePassthrough()}) | RunnableLambda(
            lambda r: compress_documents_lambda(documents=r["documents"], query=r["query"], k=10)
        )
    elif reranker == "Cross-encoder":
        model = HuggingFaceCrossEncoder(model_name=RERANKER_CROSS_ENCODER)
        compressor = CrossEncoderReranker(model=model, top_n=10)
        retrieval_agent = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    elif reranker == "ColBERT":
        colBERT = RAGPretrainedModel.from_pretrained(RERANKER_COLBERT)
        retrieval_agent = ContextualCompressionRetriever(base_compressor=colBERT.as_langchain_document_compressor(k=10), base_retriever=retriever)
    elif reranker == "Ensemble":
        # BM25
        reranker_1 = RunnableParallel({"documents": retriever, "query": RunnablePassthrough()}) | RunnableLambda(
            lambda r: compress_documents_lambda(documents=r["documents"], query=r["query"], k=10)
        )
        # Cross encoder
        compressor = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=RERANKER_CROSS_ENCODER), top_n=10)
        reranker_2 = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

        # ColBERT
        reranker_3 = ContextualCompressionRetriever(
            base_compressor=RAGPretrainedModel.from_pretrained(RERANKER_COLBERT).as_langchain_document_compressor(k=10),
            base_retriever=retriever,
        )

        retrieval_agent = EnsembleRetriever(retrievers=[reranker_1, reranker_2, reranker_3], weigths=[1 / 3, 1 / 3, 1 / 3])
    else:
        raise ValueError("This reranking method is not handled by the ChatBot or does not exist")

    # build the first part of the chain
    retriever_chain = RunnableParallel({"context": retrieval_agent, "question": RunnablePassthrough()})

    if bool_log:
        return retriever_chain | RunnableLambda(log_chain_results).bind(prompt=None, reranker=reranker)
    else:
        return retriever_chain
