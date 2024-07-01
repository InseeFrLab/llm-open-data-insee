import logging

import pandas as pd
import numpy as np

from typing import Dict, Tuple
from scipy.sparse import csr_matrix
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader

from evaluation import RetrievalConfiguration

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from evaluation.utils import build_chain_retriever_test
from config import EMB_MODEL_NAME, MODEL_NAME
from db_building import build_database_from_dataframe

## Utility function ######

"""def build_embedding_model(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False
    )"""


def recall(retrieved, relevant):
    intersection = set(retrieved) & set(relevant)
    return np.round(len(intersection) / len(relevant), 3) if len(relevant) > 0 else 0


def precision(retrieved, relevant):
    intersection = set(retrieved) & set(relevant)
    return np.round(len(intersection) / len(retrieved), 3) if len(retrieved) > 0 else 0


def hit_rate(retrieved, relevant):
    """
    Hit rate metric is equivalent to the accuracy
    """
    correct_retrieved = sum(
        1 for pred, label in zip(retrieved, relevant) if pred == label
    )
    total_retrieved = len(retrieved)
    hit_rate = correct_retrieved / total_retrieved
    return hit_rate


def mmr(retrieved, relevant):
    # compute Mean Reciprocal Rank (Order Aware Metrics)
    if relevant not in retrieved:
        mrr_score = 1 / np.inf
    else:
        rank_q = retrieved.index(relevant)
        mrr_score = 1 / (rank_q + 1)
    return mrr_score

def build_vector_database(path_data: str, config: RetrievalConfiguration) -> Chroma:
    """
    Building vector database based on a given embedding model
    """
    embedding_model_name = config.get("embedding_model_name", EMB_MODEL_NAME)

    raw_ref_database = pd.read_csv(path_data)

    vector_db = build_database_from_dataframe(
        df=raw_ref_database,
        persist_directory="./data/chroma_db",
        embedding_model_name=embedding_model_name,
        collection_name="insee_data_" + str(embedding_model_name.split("/")[-1]),
    )
    return vector_db

## Main class ###########

class RetrievalEvaluator:

    @staticmethod
    def run(
        eval_configurations: list[RetrievalConfiguration],
        eval_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, Tuple[csr_matrix, Dict, Dict, Dict]]:
        """
        Goal : Evaluate the retrieval performance of a series of configurations.
        eval_dict : dictionary containing a DataFrame with at least a question and source columns.
        """
        results = {}
        for df_name, df in eval_dict.items():
            print(df_name)
            results[df_name] = {}
            for configuration in eval_configurations:
                config_name = configuration.name
                logging.info(
                    f"Start of evaluation run for dataset: {df_name} and configuration: {config_name}"
                )
                results[df_name][config_name] = {}

                # Load ref Corpus
                vector_db = build_vector_database(
                    path_data=configuration.database_path, config=configuration
                )

                # create a retriever
                # note : define the type of search "similarity", "mmr", "similarity_score_threshold"
                # "mmr" promote diversity with 'lambda_mult': 0 (max diversity) - 1 (min diversity)
                # 'fetch_k': fetch more documents for the MMR algo
                base_retriever = vector_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": max(configuration.k_values)},
                )

                queries = [{"query": q} for q in list(df["question"])]
                """logging.info(f"   Starting to embed questions")
                query_embeddings = embedding_model.embed_documents(queries)
                logging.info(f"   The questions have been embedded")"""

                # load retriever
                retriever = build_chain_retriever_test(
                    base_retriever=base_retriever, config=configuration
                )

                # run retrieval phases
                complete_retrieved_documents = retriever.batch(
                    inputs=queries, config={"max_concurrency": 10}
                )

                all_individual_recalls = []
                all_individual_precisions = []
                for i, row in tqdm(df.iterrows()):
                    individual_recalls = []
                    individual_precisions = []
                    # q = row["question"]
                    golden_source = row.get("source_doc")
                    """
                    query_embedding = query_embeddings[i]
                    nb_retrieved = max(configuration.k_values)
                    retrieved_docs = vector_db.similarity_search_by_vector(embedding=query_embedding, 
                                                                           k=nb_retrieved)
                    """
                    retrieved_docs = complete_retrieved_documents[i]
                    retrieved_sources = [
                        doc.metadata["source"] for doc in retrieved_docs
                    ]
                    if i % 50 == 0:
                        logging.info(
                            f"      Relevant sources have been retrieved for question {i} "
                        )

                    for k in configuration.k_values:
                        if i % 50 == 0:
                            logging.info(
                                f"      Computing measures at level {k} in {configuration.k_values}  for question {i}"
                            )
                        recall_at_k = recall(retrieved_sources[:k], [golden_source])
                        precision_at_k = precision(
                            retrieved_sources[:k], [golden_source]
                        )
                        individual_recalls.append(recall_at_k)
                        individual_precisions.append(precision_at_k)

                    all_individual_recalls.append(individual_recalls)
                    all_individual_precisions.append(individual_precisions)
                # length of all_individual_recalls/precisions equals len(configuration.k_values)
                assert len(df) == len(all_individual_recalls), "nb of individuals error"
                logging.info(
                    f"      length of all_individual recalls is {len(all_individual_recalls)}"
                )
                all_k_precisions = list(zip(*all_individual_precisions))
                all_k_recalls = list(zip(*all_individual_recalls))
                # logging.info(f"      length of all_individual recalls is {len(all_individual_recalls)}")
                assert len(configuration.k_values) == len(
                    all_k_recalls
                ), "Number of ks error"
                results[df_name][config_name]["recall"] = {}
                results[df_name][config_name]["precision"] = {}
                for i, k in enumerate(configuration.k_values):
                    results[df_name][config_name]["recall"][k] = np.mean(
                        all_k_recalls[i]
                    )
                    results[df_name][config_name]["precision"][k] = np.mean(
                        all_k_precisions[i]
                    )
        return results

    @staticmethod
    def build_reference_matrix(df: pd.DataFrame) -> Tuple[csr_matrix, Dict, Dict, Dict]:
        unique_questions = pd.unique(df["question"])
        unique_sources = pd.unique(df["source_doc"])
        unique_objects = np.concatenate([unique_questions, unique_sources])
        val_to_id = {val: i for i, val in enumerate(unique_objects)}
        id_to_val = {v: k for k, v in val_to_id.items()}
        # Digitize the df
        digit_df = pd.DataFrame()
        # for questions use "standard" indices (indices relative to the whole set)
        digit_df["question"] = df["question"].replace(val_to_id, inplace=False)
        # for source docs use "relative" numeric indices ("relative" to the matrix)
        source_val_to_local_id = {
            v: local_id for local_id, v in enumerate(unique_sources)
        }
        digit_df["source_doc"] = df["source_doc"].replace(
            source_val_to_local_id, inplace=False
        )
        # Build a matrix using the indices of all the objects
        question_ids = digit_df["question"].values
        source_ids = digit_df["source_doc"].values
        X = csr_matrix(
            (
                np.ones(len(df)),
                (question_ids, source_ids),
            )
        )
        return X, id_to_val, val_to_id, source_val_to_local_id
