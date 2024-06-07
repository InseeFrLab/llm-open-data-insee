import numpy as np
import pandas as pd
from typing import Dict

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

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
