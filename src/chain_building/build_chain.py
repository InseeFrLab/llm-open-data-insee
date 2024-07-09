
# loading rerankers
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from src.reranking import compress_documents_lambda
from src.utils import format_docs

RERANKER_CROSS_ENCODER = "BAAI/bge-reranker-base"
#RERANKER_COLBERT = "bclavie/FraColBERTv2"


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
            lambda r: compress_documents_lambda(documents=r["documents"], query=r["query"], k=5)
        )
    elif reranker == "Cross-encoder":
        model = HuggingFaceCrossEncoder(model_name=RERANKER_CROSS_ENCODER)
        compressor = CrossEncoderReranker(model=model, top_n=5)
        retrieval_agent = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    elif reranker == "Ensemble":
        # BM25
        reranker_1 = RunnableParallel({"documents": retriever, "query": RunnablePassthrough()}) | RunnableLambda(
            lambda r: compress_documents_lambda(documents=r["documents"], query=r["query"], k=5)
        )
        # Cross encoder
        compressor = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=RERANKER_CROSS_ENCODER), top_n=5)
        reranker_2 = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

        retrieval_agent = EnsembleRetriever(retrievers=[reranker_1, reranker_2], weigths=[1/2, 1/2])
    else:
        raise ValueError(f"Reranking method {reranker} is not implemented.")

    # build the first part of the chain
    rag_chain_with_source = RunnableParallel({"context": retrieval_agent, "question": RunnablePassthrough()}).assign(answer=chain)

    return rag_chain_with_source


def build_chain_retriever(retriever, reranker=None):
    """
    Build a langchain chain without generation, focusing on retrieving right ressources
    """
    # Define the retrieval reranker strategy
    if reranker is None:
        retrieval_agent = retriever
    elif reranker == "BM25":
        retrieval_agent = RunnableParallel({"documents": retriever, "query": RunnablePassthrough()}) | RunnableLambda(
            lambda r: compress_documents_lambda(documents=r["documents"], query=r["query"], k=5)
        )
    elif reranker == "Cross-encoder":
        model = HuggingFaceCrossEncoder(model_name=RERANKER_CROSS_ENCODER)
        compressor = CrossEncoderReranker(model=model, top_n=10)
        retrieval_agent = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    elif reranker == "Ensemble":
        # BM25
        reranker_1 = RunnableParallel({"documents": retriever, "query": RunnablePassthrough()}) | RunnableLambda(
            lambda r: compress_documents_lambda(documents=r["documents"], query=r["query"], k=5)
        )
        # Cross encoder
        compressor = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=RERANKER_CROSS_ENCODER), top_n=5)
        reranker_2 = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

        retrieval_agent = EnsembleRetriever(retrievers=[reranker_1, reranker_2], weigths=[1/2, 1/2])
    else:
        raise ValueError("This reranking method is not handled by the ChatBot or does not exist")

    # build the first part of the chain
    retriever_chain = RunnableParallel({"context": retrieval_agent, "question": RunnablePassthrough()})

    return retriever_chain
