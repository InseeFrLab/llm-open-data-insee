"""
Create Label Studio tasks for Insee Contact data.
For now, we keep only the first part of exchanges:
question asked by the user and first response.
"""
import json
import pandas as pd
from anonymize import anonymize_insee_contact_message
from ner import ner_series
from utils import fs, create_ls_task
from constants import RAW_DATA, LS_DATA_PATH


def insee_contact_to_s3():
    """
    Start from raw Insee Contact data, anonymize
    and upload Label Studio json tasks to s3.
    """
    with fs.open(RAW_DATA) as f:
        df = pd.read_csv(f)

    df = df.fillna("")[["Exchange1", "Exchange2"]]
    # Filtering empty answers
    df = df[df["Exchange2"] != ""]

    # NER on questions and answers
    questions_ner = ner_series(df["Exchange1"]).to_list()
    answers_ner = ner_series(df["Exchange2"]).to_list()

    # Anonymized questions and answers
    anonymized_questions = []
    anonymized_answers = []
    for message, ner in zip(df["Exchange1"], questions_ner):
        anonymized_questions.append(anonymize_insee_contact_message(message, ner))
    for message, ner in zip(df["Exchange2"], answers_ner):
        anonymized_answers.append(anonymize_insee_contact_message(message, ner))

    # Json tasks creation
    for idx, (question, answer) in enumerate(zip(anonymized_questions, anonymized_answers)):
        ls_task = create_ls_task(question, answer)
        with fs.open(LS_DATA_PATH + f"{idx}.json", "w") as f:
            json.dump(ls_task, f)


if __name__ == "__main__":
    insee_contact_to_s3()