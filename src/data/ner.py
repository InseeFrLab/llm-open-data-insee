"""
Apply named entity recognition to data in order to identify:
    - names;
    - addresses;
    - email addresses;
"""

import json

import numpy as np
import pandas as pd
from transformers import TokenClassificationPipeline, pipeline

from utils import fs

ner = pipeline(
    task="ner",
    model="cmarkea/distilcamembert-base-ner",
    tokenizer="cmarkea/distilcamembert-base-ner",
    aggregation_strategy="simple",
)


def custom_ner(ner_pipeline: TokenClassificationPipeline, text: str) -> list[dict]:
    """
    Ner except return empty string if text is empty.

    Args:
        ner_pipeline (TokenClassificationPipeline): NER pipeline.
        text (str): Text.

    Returns:
        Dict: NER output.
    """
    if text is None or text == "":
        return ""
    else:
        return ner(text)


def ner_series(strings: pd.Series) -> pd.Series:
    """
    Apply named entity recognition to data in order to identify:

    - PER: personality;
    - LOC: location;
    - ORG: organization;
    - MISC: miscellaneous entities (movies title, books, etc.);
    - O: background (Outside entity).

    Args:
        strings (pd.Series): Series of strings to apply NER on.
    Returns:
        pd.Series: Series of NER outputs.
    """
    output = strings.apply(lambda x: custom_ner(ner, x))
    for sublist in output:
        for dictionary in sublist:
            for key, value in dictionary.items():
                if isinstance(value, np.float32):
                    dictionary[key] = float(value)
    return output


if __name__ == "__main__":
    # Apply NER on data from Insee Contact
    path = "projet-llm-insee-open-data/data/insee_contact/data_2019_eval.csv"
    with fs.open(path) as f:
        df = pd.read_csv(f)

    # Save NER outputs
    with fs.open(
        "projet-llm-insee-open-data/data/insee_contact/ner/data_2019_eval_exchange1_ner.json",
        "w",
    ) as fp:
        json.dump(ner_series(df["Exchange1"].fillna("")).to_list(), fp)
    with fs.open(
        "projet-llm-insee-open-data/data/insee_contact/ner/data_2019_eval_exchange2_ner.json",
        "w",
    ) as fp:
        json.dump(ner_series(df["Exchange2"].fillna("")).to_list(), fp)
