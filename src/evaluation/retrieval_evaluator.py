import logging

import pandas as pd
import numpy as np

from typing import Dict, Tuple
from scipy.sparse import csr_matrix
from tqdm import tqdm


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from evaluation import RetrievalConfiguration
from retrieval_evaluation_measures import recall, precision, hit_rate, mrr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

## Utility function ######

def build_embedding_model(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False
    )

## Main class ###########

class RetrievalEvaluator:

    @staticmethod
    def run(eval_configurations: list[RetrievalConfiguration],
            vector_db : Chroma,
            dataframe_dict: Dict[str, pd.DataFrame],
           ) -> Dict[str, Tuple[csr_matrix, Dict, Dict, Dict]]:

        """
        vector_db : Chroma Test database, make sure the questions are based on the dataframe documents
        """
        results = {}
        for df_name, df in dataframe_dict.items():
            results[df_name] = {}
            for configuration in eval_configurations:
                config_name = configuration.name
                logging.info(f"Start of evaluation run for dataset: {df_name} and configuration: {config_name}")
                results[df_name][config_name] = {}
                embedding_model = build_embedding_model(configuration.embedding_model_name)
                queries = list(df["question"])
                logging.info(f"   Starting to embed questions")
                query_embeddings = embedding_model.embed_documents(queries)
                logging.info(f"   The questions have been embedded")
                all_individual_recalls = []
                all_individual_precisions = []
                for i, row in tqdm(df.iterrows()):
                    individual_recalls = []
                    individual_precisions = []
                    # q = row["question"]
                    golden_source = row["source_doc"]
                    query_embedding = query_embeddings[i]
                    nb_retrieved = max(configuration.k_values)
                    retrieved_docs = vector_db.similarity_search_by_vector(embedding=query_embedding, 
                                                                           k=nb_retrieved)
                    retrieved_sources = [doc.metadata["source"] for doc in retrieved_docs]
                    if i % 50 == 0:
                        logging.info(f"      Relevant sources have been retrieved for question {i} ")
                    for k in configuration.k_values:
                        if i % 50 == 0:
                            logging.info(f"      Computing measures at level {k} in {configuration.k_values}  for question {i}")
                        recall_at_k = recall(retrieved_sources[:k], [golden_source])
                        precision_at_k = precision(retrieved_sources[:k], [golden_source])
                        individual_recalls.append(recall_at_k)
                        individual_precisions.append(precision_at_k)
                    all_individual_recalls.append(individual_recalls)
                    all_individual_precisions.append(individual_precisions)
                # length of all_individual_recalls/precisions equals len(configuration.k_values)
                assert len(df) == len(all_individual_recalls), "nb of individuals error"
                logging.info(f"      length of all_individual recalls is {len(all_individual_recalls)}")
                all_k_precisions = list(zip(*all_individual_precisions))
                all_k_recalls = list(zip(*all_individual_recalls))
                # logging.info(f"      length of all_individual recalls is {len(all_individual_recalls)}")
                assert len(configuration.k_values) == len(all_k_recalls), "Number of ks error"
                results[df_name][config_name]["recall"] = {}
                results[df_name][config_name]["precision"] = {}
                for i, k in enumerate(configuration.k_values):
                    results[df_name][config_name]["recall"][k] =  np.mean(all_k_recalls[i])
                    results[df_name][config_name]["precision"][k] =  np.mean(all_k_precisions[i])
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
        source_val_to_local_id = {v: local_id for local_id, v in enumerate(unique_sources) }
        digit_df["source_doc"] = df["source_doc"].replace(source_val_to_local_id, inplace=False)
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

