import json
import os
from datetime import datetime

import s3fs
from langchain_core.documents.base import Document

# S3 configuration
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
s3_fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})


def log_qa_to_s3(
    dir_s3: str,
    conversation_id: str,
    message_id: str,
    user_query: str = None,
    generated_answer: str = None,
    retrieved_documents: list[Document] = None,
    prompt_template: str = None,
    embedding_model_name: str = None,
    LLM_name: str = None,
    reranker: str = None
):
    retrieved_documents_text = [d.page_content for d in retrieved_documents]
    retrieved_documents_metadata = [d.metadata for d in retrieved_documents]

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
    target_path_s3 = os.path.join(dir_s3, today_date, f"{conversation_id}.json")

    # Log to S3
    if s3_fs.exists(target_path_s3):
        # If the conversation entry already exists, append the new Q/A
        with s3_fs.open(target_path_s3, "r") as file_in:
            existing_log = json.load(file_in)
        existing_log.update(log_entry)
        with s3_fs.open(target_path_s3, "w") as file_out:
            json.dump(existing_log, file_out, indent=4)
    else:
        # Else, create it
        with s3_fs.open(target_path_s3, "w") as file_out:
            json.dump(log_entry, file_out, indent=4)
