from collections.abc import Mapping
from typing import Any

import pandas as pd
import s3fs

from src.config import RAGConfig, load_config, minimal_argparser


def process_insee_contact_data(path: str, config: Mapping[str, Any] = vars(RAGConfig())):
    """
    Process raw Insee contact data.
    """
    fs = s3fs.S3FileSystem(endpoint_url=config["s3_endpoint_url"])

    with fs.open(path) as f:
        df = pd.read_csv(f)
    # To anonymize, we will keep only the variables Exchange1 and Exchange2
    df = df[["Exchange1", "Exchange2"]]

    # We sample an evaluation dataset of 200 conversations, which we will anonymize
    # and use to evaluate the quality of the anonymization
    df_eval = df.sample(200, random_state=42)

    # Save to s3
    with fs.open("projet-llm-insee-open-data/data/insee_contact/data_2019_eval.csv", "w") as f:
        df_eval.to_csv(f, index=False)


if __name__ == "__main__":
    path = "projet-llm-insee-open-data/data/insee_contact/data_2019.csv"
    config = load_config(minimal_argparser())["data"]
    process_insee_contact_data(path)
