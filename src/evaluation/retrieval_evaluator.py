import logging
import os
import time

import chromadb
import numpy as np
import pandas as pd
from langchain_community.vectorstores.chroma import Chroma
from scipy.sparse import csr_matrix
from tqdm import tqdm

from src.config import EMB_DEVICE, EMB_MODEL_NAME
from src.db_building import build_vector_database, reload_database_from_local_dir

from .eval_configuration import RetrievalConfiguration
from .retrieval_evaluation_measures import RetrievalEvaluationMeasure
from .utils import build_chain_reranker_test

## Utility function ##
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _build_vector_database(path_data: str, config: RetrievalConfiguration) -> Chroma:
    """
    Building vector database based on a given embedding model
    """
    embedding_model_name = config.get("embedding_model_name", EMB_MODEL_NAME)
    persist_directory = "./data/chroma_db"

    # Ensure the persist directory exists
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        logging.info(f"Created persist directory: {persist_directory}")

    client = chromadb.PersistentClient(path=persist_directory)

    collection_name = config.collection
    list_collections = [c.name for c in client.list_collections()]

    logging.info(f"collection name : {collection_name}")
    logging.info(f"The available collections : {list_collections}")

    if collection_name in list_collections:
        vector_db = reload_database_from_local_dir(
            embed_model_name=embedding_model_name,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embed_device=EMB_DEVICE,
            config=config,
        )
        logging.info("The database already exists")
    else:
        logging.info("The database will be created")
        raw_ref_database = pd.read_csv(path_data)
        vector_db = build_vector_database(
            df=raw_ref_database,
            persist_directory=persist_directory,
            embedding_model_name=embedding_model_name,
            collection_name=collection_name,
            config=config,
        )

    return vector_db


## Main class ###########


class RetrievalEvaluator:
    @staticmethod
    def run(
        eval_configurations: list[RetrievalConfiguration],
        eval_dict: dict[str, pd.DataFrame],
    ) -> dict[str, tuple[csr_matrix, dict, dict, dict]]:
        """
        Goal : Evaluate the retrieval performance of a series of configurations.
        eval_dict : dictionary containing a DataFrame with at least a question and source columns.
        """
        results = {}
        ir_measures = RetrievalEvaluationMeasure()

        for df_name, df in eval_dict.items():
            results[df_name] = {}
            for configuration in eval_configurations:
                t0 = time.time()
                config_name = configuration.name
                logging.info(
                    f"Start of evaluation run for dataset: {df_name} and configuration: {config_name}"
                )
                results[df_name][config_name] = {}

                # Load ref Corpus
                vector_db = _build_vector_database(
                    path_data=configuration.database_path, config=configuration
                )

                # create a retriever
                # note : define the type of search "similarity", "mmr", "similarity_score_threshold"
                # "mmr" promote diversity with 'lambda_mult': 0 (max diversity) - 1 (min diversity)
                # 'fetch_k': fetch more documents for the MMR algo

                # retrieve queries
                queries = [q for q in list(df["question"])]

                # choose reranker based on configuration
                reranker = build_chain_reranker_test(configuration)
                logging.info("Retriever sucessfully built")
                # embed queries
                embedded_queries = vector_db.embeddings.embed_documents(queries)
                logging.info("Queries have been embedded")

                all_individual_recalls = []
                all_individual_precisions = []
                all_individual_mrrs = []
                all_individual_ndcgs = []

                for i, row in tqdm(df.iterrows()):
                    individual_recalls = []
                    individual_precisions = []
                    individual_mrrs = []
                    individual_ndcgs = []

                    golden_source = row.get("source_doc")

                    # based retriever
                    retrieved_docs = vector_db.similarity_search_by_vector(
                        embedding=embedded_queries[i], k=max(configuration.k_values)
                    )

                    # reranker
                    retrieved_docs = reranker.invoke(
                        {"documents": retrieved_docs, "query": queries[i]}
                    )

                    retrieved_sources = [
                        doc.metadata["source"] for doc in retrieved_docs
                    ]

                    for k in configuration.k_values:
                        recall_at_k = ir_measures.recall(
                            retrieved_sources[:k], [golden_source]
                        )
                        precision_at_k = ir_measures.precision(
                            retrieved_sources[:k], [golden_source]
                        )
                        mrr_at_k = ir_measures.mrr(
                            retrieved_sources[:k], [golden_source]
                        )
                        ndcg_at_k = ir_measures.ndcg(
                            retrieved_sources[:k], [golden_source]
                        )

                        individual_recalls.append(recall_at_k)
                        individual_precisions.append(precision_at_k)
                        individual_mrrs.append(mrr_at_k)
                        individual_ndcgs.append(ndcg_at_k)

                    all_individual_recalls.append(individual_recalls)
                    all_individual_precisions.append(individual_precisions)
                    all_individual_mrrs.append(individual_mrrs)
                    all_individual_ndcgs.append(individual_ndcgs)

                # length of all_individual_recalls/precisions equals len(configuration.k_values)
                assert len(df) == len(all_individual_recalls), "nb of individuals error"
                logging.info(
                    f"      length of all_individual recalls is {len(all_individual_recalls)}"
                )
                all_k_precisions = list(zip(*all_individual_precisions, strict=False))
                all_k_recalls = list(zip(*all_individual_recalls, strict=False))
                all_k_mrrs = list(zip(*all_individual_mrrs, strict=False))
                all_k_ndcgs = list(zip(*all_individual_ndcgs, strict=False))

                assert len(configuration.k_values) == len(
                    all_k_recalls
                ), "Number of ks error"
                results[df_name][config_name]["recall"] = {}
                results[df_name][config_name]["precision"] = {}
                results[df_name][config_name]["mrr"] = {}
                results[df_name][config_name]["ndcg"] = {}

                for i, k in enumerate(configuration.k_values):
                    results[df_name][config_name]["recall"][k] = np.mean(
                        all_k_recalls[i]
                    )
                    results[df_name][config_name]["precision"][k] = np.mean(
                        all_k_precisions[i]
                    )
                    results[df_name][config_name]["mrr"][k] = np.mean(all_k_mrrs[i])
                    results[df_name][config_name]["ndcg"][k] = np.mean(all_k_ndcgs[i])

                results[df_name][config_name]["runtime"] = time.time() - t0

        return results

    @staticmethod
    def build_reference_matrix(df: pd.DataFrame) -> tuple[csr_matrix, dict, dict, dict]:
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
