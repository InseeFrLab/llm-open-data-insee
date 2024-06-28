import os
import datetime
import json

import s3fs
from langchain_core.documents.base import Document


# S3 configuration
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
s3_fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})


def log_conversation_to_s3(
    conv_id: str,
    dir_s3: str,
    user_query: str = None,
    retrieved_documents: list[Document] = None,
    prompt_template: str = None,
    generated_answer: str = None,
    embedding_model_name: str = None,
    LLM_name: str = None,
    reranker: str = None
):
    retrieved_documents_text = [d.page_content for d in retrieved_documents]
    retrieved_documents_metadata = [d.metadata for d in retrieved_documents]

    msg_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_entry = {
        "user_query": user_query,
        "retrieved_docs_text": retrieved_documents_text,
        "prompt": prompt_template,
        "generated_answer": generated_answer,
        "embedding_model": embedding_model_name,
        "llm": LLM_name,
        "reranker": reranker,
        "retrieved_doc_metadata": retrieved_documents_metadata
    }

    target_path_s3 = os.path.join(dir_s3, conv_id, f"{msg_timestamp}.json")
    with s3_fs.open(target_path_s3, "w") as file_out:
        json.dump(log_entry, file_out, indent=4)
