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

#langchain packages
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

#loading rerankers
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from typing import Any, List, Optional, Sequence, Dict
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from ragatouille import RAGPretrainedModel 
from langchain.schema import Document
from pydantic import BaseModel, Field


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

"""
def rerank_with_ColBERT(reranker, query : str, retrieved_docs : List[LangchainDocument], filter_k : int ):
    relevant_docs = [doc.page_content for doc in retrieved_docs]  # keep only text
    reranked_docs = reranker.rerank(query=query, documents=relevant_docs, k=filter_k)
    content_to_doc = {doc.page_content: doc for doc in retrieved_docs if isinstance(doc.page_content, str)}
    return [content_to_doc[doc["content"]] for doc in reranked_docs]


def rerank_with_BM25(model_class, query : str, retrieved_docs : List[LangchainDocument], filter_k : int):
    relevant_docs = [doc.page_content for doc in retrieved_docs]
    bm25 = model_class(relevant_docs)
    tokenized_query = query.split()
    return bm25.get_top_n(tokenized_query, retrieved_docs, n=filter_k)"""


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

# Define the compression function
def compress_BM25_lambda(documents: Sequence[Document], query: str, k: int = 5, **kwargs: Dict[str, Any]) -> Sequence[Document]:
    """Compress retrieved documents given the query context."""

    # Initialize the retriever with the documents
    retriever = BM25Retriever.from_documents(documents, k=k, **kwargs)
    return retriever.get_relevant_documents(query)

# Define the compression function using Metadata
def compress_Metadata_lambda(documents: Sequence[LangchainDocument], query: str, config: Dict) -> Sequence[LangchainDocument]:
    
    rerank_k = config.get("rerank_k", 5)
    metadata_field = config.get("use_metadata", None)

    if metadata_field is not None: 
        new_data = []
        for doc in documents:
            meta = doc.metadata
            if metadata_field in meta and len(meta[metadata_field]) > 0:
                page_content = meta[metadata_field]
            else:
                page_content = doc.page_content
            
            new_data.append(
                LangchainDocument(
                    page_content=page_content, 
                    metadata={"source": meta.get("source", "unknown")}
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


def choosing_reranker(base_retriever, reranker_type , reranker_name, rerank_k): 
    """ 
    Return a langchain compatible augmented retriever. 
    """
    if reranker_type is None:
        retrieval_agent = base_retriever 

    elif reranker_type == "BM25":
        retrieval_agent = (
            RunnableParallel({"documents": base_retriever, "query": RunnablePassthrough()})
            | RunnableLambda(lambda r: compress_BM25_lambda(documents=r["documents"], query=r["query"], k=rerank_k))
        )
    elif reranker_type == "Cross-encoder":
        model = HuggingFaceCrossEncoder(model_name=reranker_name)
        compressor = CrossEncoderReranker(model=model, top_n=rerank_k)
        retrieval_agent = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    
    elif reranker_type == "ColBERT":
        colBERT = RAGPretrainedModel.from_pretrained(reranker_name)
        retrieval_agent = ContextualCompressionRetriever(
            base_compressor=colBERT.as_langchain_document_compressor(k=rerank_k), base_retriever=base_retriever
        )
    return retrieval_agent


def build_chain_retriever(base_retriever, config = Dict):
    """ 
    Build a langchain chain without generation, focusing on retrieving right documents 
    Customizable Implementation for testing different retrieving strategy. 
    """
    # {"reranker_type" : ..., "reranker_name": ..., "rerank_k" : < "nb_retrieved", "param_ensemble" : [{"reranker_type" : ..., "reranker_name" : ..., "reranker_weight"} )]}

    reranker_type = config.get("reranker_type", None)
    reranker_name = config.get("reranker_name", None)
    rerank_k = config.get("rerank_k", 5)
    
    if reranker_type in ["BM25","Cross-encoder","ColBERT"]:
        retrieval_agent = choosing_reranker(base_retriever, reranker_type , reranker_name, rerank_k)

    elif reranker_type == "Ensemble":

        weigths = []
        base_retrievers = []
        for config in config.get("param_ensemble", []):
            (r_type, r_name, r_w) = config.values()
            reranker_model = choosing_reranker(base_retriever, r_type , r_name, rerank_k)

            #add model to the lis of rerankers
            base_retrievers.append(reranker_model)
            weigths.append(r_w)
        
        #build ensemble model

        weights = [1/len(weights) for _ in weights] if np.sum(weights) !=1 else weights
        
        retrieval_agent = EnsembleRetriever(
            base_retrievers=base_retrievers, 
            weigths=weights
        )

    return retrieval_agent