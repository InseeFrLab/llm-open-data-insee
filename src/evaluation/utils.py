import numpy as np
import pandas as pd
from typing import Dict

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

#Langchain 
from langchain.docstore.document import Document as LangchainDocument

#reranking
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from ragatouille import RAGPretrainedModel

def use_sbert_retrieval_evaluator(df: pd.DataFrame, 
                                  model: SentenceTransformer) -> Dict:
    """
    Usage:
       df = pd.read_csv("retrieval_evaluation_Phi-3-mini-128k-instruct.csv")
       model = SentenceTransformer('all-mpnet-base-v2')
       use_sbert_retrieval_evaluator(df, model)
    """
    unique_questions = pd.unique(df["question"]) 
    unique_sources = pd.unique(df["source_doc"]) 
    queries = { str(i): q for i, q in enumerate(unique_questions)}
    rev_queries = { q : i for i, q in queries.items()}
    corpus = { str(i): d for i, d in enumerate(unique_sources)}
    rev_corpus = { d : i for i, d in corpus.items()}
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
        name="Test"
    )
    return ir_evaluator(model)


def rerank_with_ColBERT(reranker, query : str, retrieved_docs : List[LangchainDocument], filter_k : int ):
    relevant_docs = [doc.page_content for doc in retrieved_docs]  # keep only text
    reranked_docs = reranker.rerank(query=query, documents=relevant_docs, k=filter_k)
    content_to_doc = {doc.page_content: doc for doc in retrieved_docs if isinstance(doc.page_content, str)}
    return [content_to_doc[doc["content"]] for doc in reranked_docs]


def rerank_with_BM25(model_class, query : str, retrieved_docs : List[LangchainDocument], filter_k : int):
    relevant_docs = [doc.page_content for doc in retrieved_docs]
    bm25 = model_class(relevant_docs)
    tokenized_query = query.split()
    return bm25.get_top_n(tokenized_query, retrieved_docs, n=filter_k)


def rerank_with_metadata(reranker, query : str, retrieved_docs : List[LangchainDocument], filter_k : int , params : Dict):
    """
    note if the metadata is missing we use a "content" information (always exists) as a fallback  
    """
    new_data = []
    for doc in retrieved_docs:
        metadata_field = params.get("use_metadata")
        if metadata_field in doc.metadata:
            page_content = doc.metadata[metadata_field]
        else:
            page_content = doc.page_content
        
        new_data.append(
            LangchainDocument(
                page_content = page_content, 
                metadata = {"source": doc.metadata.get("source", "unknown")}
                )
            )
    #load reranker 
    new_retrieved_docs = rerank_with_ColBERT(reranker=reranker, query=query, retrieved_docs = new_data, filter_k=filter_k)

    source_to_doc_map = {doc.metadata.get("source", "unknown"): doc for doc in retrieved_docs}
    #reorder the original retrieved documents based on thee new reranked docs
    reordered_docs = [source_to_doc_map[new_doc.metadata["source"]] for new_doc in new_retrieved_docs if new_doc.metadata["source"] in source_to_doc_map]

    return reordered_docs