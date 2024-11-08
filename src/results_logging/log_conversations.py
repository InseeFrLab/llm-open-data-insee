import json
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import s3fs
from langchain_core.documents.base import Document

from src.config import default_config


def log_qa_to_s3(
    filesystem: s3fs.S3FileSystem,
    thread_id: str,
    message_id: str,
    user_query: str | None = None,
    generated_answer: str | None = None,
    retrieved_documents: list[Document] | None = None,
    prompt_template: str | None = None,
    embedding_model_name: str | None = None,
    LLM_name: str | None = None,
    reranker: str | None = None,
    config: Mapping[str, Any] = default_config,
):
    retrieved_documents_text = [d.page_content for d in retrieved_documents] if retrieved_documents else None
    retrieved_documents_metadata = [d.metadata for d in retrieved_documents] if retrieved_documents else None

    log_entry = {
        message_id: {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": user_query,
            "generated_answer": generated_answer,
            "retrieved_docs_text": retrieved_documents_text,
            "retrieved_docs_metadata": retrieved_documents_metadata,
            "prompt": prompt_template,
            "embedding_model": embedding_model_name,
            "llm": LLM_name,
            "reranker": reranker,
        }
    }

    today_date = datetime.now().strftime("%Y-%m-%d")
    target_path_s3 = os.path.join(config["s3_bucket"], "data", "chatbot_logs", today_date, f"{thread_id}.json")

    # Log to S3
    if filesystem.exists(target_path_s3):
        # If the conversation entry already exists, append the new Q/A
        with filesystem.open(target_path_s3, "r") as file_in:
            existing_log = json.load(file_in)
        existing_log.update(log_entry)
        with filesystem.open(target_path_s3, "w") as file_out:
            json.dump(existing_log, file_out, indent=4)
    else:
        # Else, create it
        with filesystem.open(target_path_s3, "w") as file_out:
            json.dump(log_entry, file_out, indent=4)


def log_feedback_to_s3(
    filesystem: s3fs.S3FileSystem,
    thread_id: str,
    message_id: str,
    feedback_value: int,
    feedback_comment: str | None = None,
    config: Mapping[str, Any] = default_config,
):
    today_date = datetime.now().strftime("%Y-%m-%d")
    target_path_s3 = os.path.join(config["s3_bucket"], "data", "chatbot_logs", today_date, f"{thread_id}.json")

    # Add feedback to existing log
    with filesystem.open(target_path_s3, "r") as file_in:
        existing_log = json.load(file_in)
    existing_log[message_id]["feedback_value"] = feedback_value
    existing_log[message_id]["feedback_comment"] = feedback_comment

    # Push to S3
    with filesystem.open(target_path_s3, "w") as file_out:
        json.dump(existing_log, file_out, indent=4)
