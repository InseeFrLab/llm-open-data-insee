import json

import pandas as pd
import s3fs
from anonymize import anonymize_insee_contact_message
from constants import LS_DATA_PATH, RAW_DATA
from ner import ner_series


def process_insee_contact_data(config, path: str):
    """
    Process raw Insee contact data.
    """
    fs = s3fs.S3FileSystem(endpoint_url=config.s3_endpoint_url)

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


def create_ls_task(question: str, answer: str) -> dict[str, dict[str, str]]:
    """
    Create Label Studio task .json for a question/answer pair.

    Args:
        question (str): Question.
        answer (str): Answer.

    Returns:
        Dict[str, str]: Label Studio json task.
    """
    return {
        "data": {
            "question": question,
            "answer": answer,
        }
    }


def insee_contact_to_s3(config):
    """
    Start from raw Insee Contact data, anonymize
    and upload Label Studio json tasks to s3.
    """
    fs = s3fs.S3FileSystem(endpoint_url=config.s3_endpoint_url)
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
    for message, ner in zip(df["Exchange1"], questions_ner, strict=False):
        anonymized_questions.append(anonymize_insee_contact_message(message, ner))
    for message, ner in zip(df["Exchange2"], answers_ner, strict=False):
        anonymized_answers.append(anonymize_insee_contact_message(message, ner))

    # Json tasks creation
    for idx, (question, answer) in enumerate(zip(anonymized_questions, anonymized_answers, strict=False)):
        ls_task = create_ls_task(question, answer)
        with fs.open(LS_DATA_PATH + f"{idx}.json", "w") as f:
            json.dump(ls_task, f)


if __name__ == "__main__":
    process_insee_contact_data("projet-llm-insee-open-data/data/insee_contact/data_2019.csv")
