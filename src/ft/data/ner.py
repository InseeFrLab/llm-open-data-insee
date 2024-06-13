"""
Apply named entity recognition to data in order to identify:
    - names;
    - addresses;
    - email addresses;
"""
import json
import s3fs
import os
import numpy as np
import pandas as pd
from transformers import pipeline


def custom_ner(text: str):
    """
    Ner except return empty string if text is empty.

    Args:
        text (str): Text.
    """
    ner = pipeline(
        task='ner',
        model="cmarkea/distilcamembert-base-ner",
        tokenizer="cmarkea/distilcamembert-base-ner",
        aggregation_strategy="simple"
    )

    if (text == "") or (text is None):
        return ""
    else:
        return ner(text)


def ner_series(strings: pd.Series):
    """
    Apply named entity recognition to data in order to identify:

    - PER: personality;
    - LOC: location;
    - ORG: organization;
    - MISC: miscellaneous entities (movies title, books, etc.);
    - O: background (Outside entity).
    """
    output = strings.apply(lambda x: custom_ner(x))
    for sublist in output:
        for dictionary in sublist:
            for key, value in dictionary.items():
                if isinstance(value, np.float32):
                    dictionary[key] = float(value)
    return output


if __name__ == "__main__":
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + os.environ["AWS_S3_ENDPOINT"]}
    )

    path = "projet-llm-insee-open-data/data/insee_contact/data_2019_eval.csv"
    with fs.open(path) as f:
        df = pd.read_csv(f)

    with fs.open('projet-llm-insee-open-data/data/insee_contact/ner/data_2019_eval_exchange1_ner.json', 'w') as fp:
        json.dump(ner_series(df["Exchange1"].fillna("")).to_list(), fp)
    with fs.open('projet-llm-insee-open-data/data/insee_contact/ner/data_2019_eval_exchange2_ner.json', 'w') as fp:
        json.dump(ner_series(df["Exchange2"].fillna("")).to_list(), fp)
