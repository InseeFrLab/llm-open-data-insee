"""
Create an evaluation dataset from Label Studio annotations stored
on S3. An observation of this data set consists in a question/answer
pair, along with a (potentially empty) list of URLs.
"""

from pathlib import Path
import json
import pandas as pd
from utils import fs
from constants import LS_ANNOTATIONS_PATH


def create_insee_contact_eval_dataset():
    questions = []
    answers = []
    urls = []
    for path in fs.listdir(LS_ANNOTATIONS_PATH):
        with fs.open(path["Key"], "rb") as f:
            if Path(path["Key"]).stem == ".keep":
                continue
            annotation_data = json.load(f)
            question = annotation_data["task"]["data"]["question"]
            answer = annotation_data["task"]["data"]["answer"]

            keep_pair = False
            entry_urls = []
            # Parse annotations
            for result_element in annotation_data["result"]:
                if result_element["from_name"] == "keep_pair":
                    if "O" in result_element["value"]["choices"]:
                        keep_pair = True
                if result_element["from_name"] == "urls":
                    entry_urls += result_element["value"]["text"]
                if result_element["from_name"] == "anon_question":
                    question = result_element["value"]["text"][0]
            if keep_pair:
                questions.append(question)
                answers.append(answer)
                urls.append("|".join(entry_urls))

    with fs.open(
        "projet-llm-insee-open-data/data/eval_data/eval_dataset_insee_contact.csv", "w"
    ) as f:
        pd.DataFrame({"questions": questions, "answers": answers, "urls": urls}).to_csv(
            f, index=False
        )


if __name__ == "__main__":
    create_insee_contact_eval_dataset()
