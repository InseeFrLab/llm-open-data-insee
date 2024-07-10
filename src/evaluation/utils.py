from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from typing import Any, TypeVar

import numpy as np

# from sentence_transformers import SentenceTransformer
# from sentence_transformers.evaluation import InformationRetrievalEvaluator
# Langchain
from langchain.docstore.document import Document as LangchainDocument

# loading rerankers
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever

# langchain packages
from langchain_core.runnables import (
    RunnableLambda,
)

# reranking
from ragatouille import RAGPretrainedModel

# evaluation
from evaluation import RetrievalConfiguration

T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


def aggregate_ensemble_reranker(reranker_outputs: dict):
    return list(reranker_outputs.values())


def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
    """Yield unique elements of an iterable based on a key function.

    Args:
        iterable: The iterable to filter.
        key: A function that returns a hashable key for each element.

    Yields:
        Unique elements of the iterable based on the key function.
    """
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e


def weighted_reciprocal_rank(doc_lists: list[list[Document]], weights, c=60) -> list[Document]:
    """
    This function comes from Langchain documentation.
    Perform weighted Reciprocal Rank Fusion on multiple rank lists.
    You can find more details about RRF here:
    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

    Args:
        doc_lists: A list of rank lists, where each rank list contains unique items.
        c: A constant added to the rank, controlling the balance between the importance
        of high-ranked items and the consideration given to lower-ranked items.
        Default is 60.
        weights: given weights to rerankers, sum to 1

    Returns:
        list: The final aggregated list of items sorted by their weighted RRF
                scores in descending order.
    """
    from collections import defaultdict
    from itertools import chain

    if len(doc_lists) != len(weights):
        raise ValueError("Number of rank lists must be equal to the number of weights.")

    # Associate each doc's content with its RRF score for later sorting by it
    # Duplicated contents across retrievers are collapsed & scored cumulatively
    rrf_score: dict[str, float] = defaultdict(float)
    for doc_list, weight in zip(doc_lists, weights, strict=True):
        for rank, doc in enumerate(doc_list, start=1):
            rrf_score[doc.page_content] += weight / (rank + c)

    # Docs are deduplicated by their contents then sorted by their scores
    all_docs = chain.from_iterable(doc_lists)
    sorted_docs = sorted(
        unique_by_key(all_docs, lambda doc: doc.page_content),
        reverse=True,
        key=lambda doc: rrf_score[doc.page_content],
    )
    return sorted_docs


"""
def use_sbert_retrieval_evaluator(df: pd.DataFrame, model: SentenceTransformer) -> Dict:

    Usage:
       df = pd.read_csv("retrieval_evaluation_Phi-3-mini-128k-instruct.csv")
       model = SentenceTransformer('all-mpnet-base-v2')
       use_sbert_retrieval_evaluator(df, model)

    unique_questions = pd.unique(df["question"])
    unique_sources = pd.unique(df["source_doc"])
    queries = {str(i): q for i, q in enumerate(unique_questions)}
    rev_queries = {q: i for i, q in queries.items()}
    corpus = {str(i): d for i, d in enumerate(unique_sources)}
    rev_corpus = {d: i for i, d in corpus.items()}
    edges_df = df[["question", "source_doc"]]
    relevant_docs = {}
    for _, row in edges_df.iterrows():
        x, y = rev_queries[row["question"]], rev_corpus[row["source_doc"]]
        if x not in result_dict:
            relevant_docs[x] = set()
        relevant_docs[x].add(y)
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        # queries entries like: '521': 'Quels sont...'
        corpus=corpus,
        # dict entries like: '22708': 'https://...'
        relevant_docs=relevant_docs,
        # dict entries like:  '521': {'22708', '29976'}
        name="Test",
    )
    return ir_evaluator(model)"""


# Define the compression function
def compress_BM25_lambda(documents: Sequence[Document], query: str, k: int = 5, **kwargs: dict[str, Any]) -> Sequence[Document]:
    """Compress retrieved documents given the query context."""

    # Initialize the retriever with the documents
    retriever = BM25Retriever.from_documents(documents, k=k, **kwargs)
    return retriever.get_relevant_documents(query)


# Define the compression function using Metadata
def compress_metadata_lambda(documents: Sequence[LangchainDocument], query: str, config: dict) -> Sequence[LangchainDocument]:
    rerank_k = config.get("rerank_k", len(documents))
    metadata_field = config.get("use_metadata")

    if metadata_field is not None:
        new_data = []
        for doc in documents:
            meta = doc.metadata
            page_content = meta[metadata_field] if metadata_field in meta and len(meta[metadata_field]) > 0 else doc.page_content
            new_data.append(
                LangchainDocument(
                    page_content=page_content,
                    metadata={"source": meta.get("source", "unknown")},
                )
            )

        # Use BM25 to rerank the new data
        new_retrieved_docs = compress_BM25_lambda(documents=new_data, query=query, k=rerank_k)

        # Map sources to original documents
        source_to_doc_map = {doc.metadata.get("source", "unknown"): doc for doc in documents}

        # Return the reranked documents based on the source mapping
        return [source_to_doc_map[new_doc.metadata["source"]] for new_doc in new_retrieved_docs if new_doc.metadata["source"] in source_to_doc_map]
    else:
        # If no metadata field specified, return top k documents
        return documents[:rerank_k]


def choosing_reranker_test(config: dict):
    """
    Return a langchain compatible augmented retriever.
    Should take as input List[Document]
    """
    reranker_type, reranker_name, rerank_k = (
        config.get("reranker_type"),
        config.get("reranker_name"),
        config.get("rerank_k", max(config.k_values)),
    )

    if reranker_type is None:
        retrieval_agent = RunnableLambda(lambda r: r["documents"])
    elif reranker_type == "BM25":
        # need to format {"documents" : ..., "query" : ...}
        retrieval_agent = RunnableLambda(lambda r: compress_BM25_lambda(documents=r["documents"], query=r["query"], k=rerank_k))
    elif reranker_type == "Cross-encoder":
        # need to format {"documents" : ..., "query" : ...}
        model = HuggingFaceCrossEncoder(model_name=reranker_name, model_kwargs={"device": "cuda"})
        compressor = CrossEncoderReranker(model=model, top_n=rerank_k)
        retrieval_agent = RunnableLambda(func=lambda inputs: compressor.compress_documents(documents=inputs["documents"], query=inputs["query"]))
    elif reranker_type == "ColBERT":
        # need to format {"documents" : ..., "query" : ...}
        colBERT = RAGPretrainedModel.from_pretrained(reranker_name)
        compressor = colBERT.as_langchain_document_compressor(k=rerank_k)
        retrieval_agent = RunnableLambda(func=lambda r: compressor.compress_documents(documents=r["documents"], query=r["query"], top_n=rerank_k))
    elif reranker_type == "Metadata":
        # need to format {"documents" : ..., "query" : ...}
        retrieval_agent = RunnableLambda(func=lambda r: compress_metadata_lambda(documents=r["documents"], query=r["query"], config=config))
    return retrieval_agent


def build_chain_reranker_test(config=RetrievalConfiguration):
    """
    Build a langchain chain without generation, focusing on retrieving right documents
    Customizable Implementation for testing different retrieving strategy.

    Ex :
        config = {"reranker_type" : ...,
                    "reranker_name": ...,
                    "rerank_k" : ... ,
                    "param_ensemble" : [{"reranker_type" : ..., "reranker_name" : ..., "reranker_weight": ...}, ...],
                    "use_metadata" : ... ,
                 }
    """
    # {"reranker_type" : ..., "reranker_name": ..., "rerank_k" : < "nb_retrieved", "param_ensemble" : 
    # [{"reranker_type" : ..., "reranker_name" : ..., "reranker_weight"} )]}

    reranker_type = config.get("reranker_type")

    if reranker_type in [None, "BM25", "Cross-encoder", "ColBERT", "Metadata"]:
        retrieval_agent = choosing_reranker_test(config=config)
    elif reranker_type == "Ensemble":
        weights = []
        results = {}

        copy_config = config.copy()
        for i, config_ind in enumerate(config.get("param_ensemble")):
            (r_type, r_name, r_w) = config_ind.values()

            copy_config = config.copy()
            copy_config.reranker_type = r_type
            copy_config.reranker_name = r_name

            reranker_model = choosing_reranker_test(config=copy_config)

            # add model to the lis of rerankers

            results[f"model_{i}"] = reranker_model
            weights.append(r_w)

        weights = [1 / len(weights) for _ in weights] if np.sum(weights) != 1 else weights  # uniform weights.

        retrieval_agent = (
            results
            | RunnableLambda(func=lambda d: aggregate_ensemble_reranker(d))
            | RunnableLambda(func=lambda d: weighted_reciprocal_rank(doc_lists=d, weights=weights))
        )

    return retrieval_agent
