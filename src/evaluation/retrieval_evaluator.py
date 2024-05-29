import pandas as pd
import numpy as np

from typing import Dict, Tuple
from scipy.sparse import csr_matrix

from evaluation import RetrievalConfiguration

class RetrievalEvaluator:

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

    @staticmethod
    def run(eval_configurations: list[RetrievalConfiguration],
            dataframe_dict: Dict[str, pd.DataFrame],
           ) -> Dict[str, Tuple[csr_matrix, Dict, Dict, Dict]]:
        reference_matrices = {}
        for k in dataframe_dict:
            df = dataframe_dict[k]
            reference_matrices[k] = RetrievalEvaluator.build_reference_matrix(df)
        return reference_matrices
    
