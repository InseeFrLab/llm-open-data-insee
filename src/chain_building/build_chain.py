from collections.abc import Sequence
from typing import Any

# loading rerankers
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

RERANKER_CROSS_ENCODER = "BAAI/bge-reranker-base"
# RERANKER_COLBERT = "bclavie/FraColBERTv2"


def format_docs(docs: list):
    return "\n\n".join(
        [
            f"Doc {i + 1}:\nTitle: {doc.metadata["title"]}\nContent:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ]
    )


# Define the compression function
def compress_documents_lambda(
    documents: Sequence[Document], query: str, k: int = 5, **kwargs: dict[str, Any]
) -> Sequence[Document]:
    """Compress retrieved documents given the query context."""

    # Initialize the retriever with the documents
    retriever = BM25Retriever.from_documents(documents, k=k, **kwargs)
    return retriever.get_relevant_documents(query)


# BUILD CHAIN ----------------------------------


def build_chain(
    retriever, prompt: str,
    llm=None, bool_log: bool = False,
    reranker: str = None,
    number_candidates_reranking: int = 10
):
    """
    Build a LLM chain based on Langchain package and INSEE data
    """

    if reranker not in [
        None, "BM25", "Cross-encoder", "Ensemble"
    ]:
        raise ValueError(
            f"Invalid reranker: {reranker}. Accepted values are: {', '.join(accepted_rerankers)}"
        )

    # Define the retrieval reranker strategy
    if reranker is None:
        retrieval_agent = retriever
    elif reranker == "BM25":
        retrieval_agent = RunnableParallel(
            {"documents": retriever, "query": RunnablePassthrough()}
        ) | RunnableLambda(
            lambda r: compress_documents_lambda(
                documents=r["documents"], query=r["query"], k=number_candidates_reranking
            )
        )
    elif reranker == "Cross-encoder":
        model = HuggingFaceCrossEncoder(model_name=RERANKER_CROSS_ENCODER)
        compressor = CrossEncoderReranker(model=model, top_n=number_candidates_reranking)
        retrieval_agent = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
    elif reranker == "Ensemble":
        # BM25
        reranker_1 = RunnableParallel(
            {"documents": retriever, "query": RunnablePassthrough()}
        ) | RunnableLambda(
            lambda r: compress_documents_lambda(
                documents=r["documents"], query=r["query"], k=number_candidates_reranking
            )
        )
        # Cross encoder
        compressor = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name=RERANKER_CROSS_ENCODER),
            top_n=number_candidates_reranking,
        )
        reranker_2 = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        retrieval_agent = EnsembleRetriever(
            retrievers=[reranker_1, reranker_2], weigths=[1 / 2, 1 / 2]
        )
    else:
        raise ValueError(f"Reranking method {reranker} is not implemented.")

    if llm is not None:
        # Create a Langchain LLM Chain
        chain = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )
        rag_chain_with_source = RunnableParallel(
            {"context": retrieval_agent, "question": RunnablePassthrough()}
        ).assign(answer=chain)
    else:
        # retriever mode
        rag_chain_with_source = RunnableParallel(
            {"context": retrieval_agent, "question": RunnablePassthrough()}
        )

    return rag_chain_with_source
