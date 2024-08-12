"""
Script to anonymize Insee Contact data.
"""

import json
import re

import pandas as pd
from utils import fs


def detect_email_signature(message: str, message_ner: list[dict]):
    """
    Returns first position of email signature in message.
    For now take a `message` string as input and returns
    the message without a signature identified as all text
    after the keyword "cordialement"
    """
    # TODO: use the `message_ner` argument
    search_term = "cordialement"
    lines = message.split("\n")
    signature_line_index = None
    signature_position_in_line = None

    # Find the line and position of "cordialement"
    for i, line in enumerate(lines):
        position = line.lower().find(search_term)
        if position != -1:
            signature_line_index = i
            signature_position_in_line = position + len(search_term)
            break

    if signature_line_index is None:
        # Return -1 or some indication that the signature was not found
        return -1

    # Calculate the character index of the first character after "cordialement"
    char_index = sum(len(line) + 1 for line in lines[:signature_line_index])
    char_index += signature_position_in_line
    return char_index


def add_signature_key_to_ner(message: str, message_ner: list[dict]):
    """
    Adds a key 'signature' to the named entities in the message
    which belong to the signature: entities after which there is no
    text content which is not an entity.

    Args:
        message (str): Message.
        message_ner (List[Dict]): Message NER.
    """
    # Init at False
    for entity in message_ner:
        entity["signature"] = False
    # Loop from end
    nchar_left = len(message)
    for entity in message_ner[::-1]:
        if entity["end"] == nchar_left - 1:
            entity["signature"] = True
            nchar_left = entity["start"]
        else:
            entity["signature"] = False
            return
    return


def anonymize_insee_contact_message(message: str, message_ner: list[dict]) -> str:
    """
    Anonymize a message given a NER output. `message_ner` is a
    list of dictionaries, each dictionary contains the named entities
    with keys 'entity_group', 'score', 'word', 'start', 'end'.

    Args:
        message (str): Message.
        message_ner (List[Dict]): Message NER.
    Returns:
        str: Anonymized message.
    """
    add_signature_key_to_ner(message, message_ner)

    # Identification of names and signature tokens
    # Replace names with the token [PER]
    # Replace signature entities with a token
    for dictionary in message_ner:
        if dictionary["entity_group"] == "PER":
            message = message.replace(dictionary["word"], "[PER]")
        elif dictionary["signature"]:
            message = message.replace(
                dictionary["word"], f"[{dictionary['entity_group']}]"
            )

    # Identification of email addresses
    email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    # Replace email addresses with the token [EMAIL]
    message = re.sub(email_regex, "[EMAIL]", message)

    # Telephone numbers
    # There can be spaces every two digits
    # There can be dashes or dots every two digits
    phone_regex = r"\b0[1-9][-. ]?([0-9]{2}[-. ]?){4}\b"
    message = re.sub(phone_regex, "[PHONE]", message)

    # Remove everything after "-----Message d'origine-----" as this indicates
    # an email is being forwarded
    message = message.split("-----Message d'origine-----")[0]

    # Remove email signature
    signature_index = detect_email_signature(message, message_ner)
    if signature_index != -1:
        message = message[:signature_index]

    return message


if __name__ == "__main__":
    # Anonymize evaluation data and export it to a .csv file
    # for manual evaluation
    path = "projet-llm-insee-open-data/data/insee_contact/data_2019_eval.csv"
    with fs.open(path) as f:
        df = pd.read_csv(f)
    df = df.fillna("")

    # Load NER data
    with fs.open(
        "projet-llm-insee-open-data/data/insee_contact/ner/data_2019_eval_exchange1_ner.json",
        "rb",
    ) as f:
        exchange1_ner = json.load(f)
    with fs.open(
        "projet-llm-insee-open-data/data/insee_contact/ner/data_2019_eval_exchange2_ner.json",
        "rb",
    ) as f:
        exchange2_ner = json.load(f)

    # Anonymize exchanges
    anonymized_exchange1 = []
    anonymized_exchange2 = []
    for message, ner in zip(df["Exchange1"], exchange1_ner, strict=False):
        anonymized_exchange1.append(anonymize_insee_contact_message(message, ner))
    for message, ner in zip(df["Exchange2"], exchange2_ner, strict=False):
        anonymized_exchange2.append(anonymize_insee_contact_message(message, ner))

    # Export anonymized questions along with original ones for evaluation
    eval_df_exchange_1 = pd.DataFrame(
        {
            "raw": df["Exchange1"],
            "anonymized": anonymized_exchange1,
        }
    )
    with fs.open(
        "projet-llm-insee-open-data/data/insee_contact/data_2019_eval_exchange1.csv",
        "w",
    ) as f:
        eval_df_exchange_1.to_csv(f, index=False)
    # Export anonymized answers along with original ones for evaluation
    eval_df_exchange_2 = pd.DataFrame(
        {
            "raw": df["Exchange2"],
            "anonymized": anonymized_exchange2,
        }
    )
    with fs.open(
        "projet-llm-insee-open-data/data/insee_contact/data_2019_eval_exchange2.csv",
        "w",
    ) as f:
        eval_df_exchange_2.to_csv(f, index=False)
