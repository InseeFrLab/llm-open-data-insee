from datetime import datetime
import logging

from config import MODEL_NAME, EMB_MODEL_NAME


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def log_chain_results(result, prompt, reranker):
    """
    Logs interaction details into a JSON file and returns the original result.

    Args:
    result (dict): The result from the RAG pipeline containing question, answer, and context.
    prompt (object): The prompt template used.
    reranker (str): The reranker used in the pipeline.

    Returns:
    dict: The original result.
    """
    # Extracting necessary details from the result
    user_query = result.get("question", None)
    generated_answer = result.get("answer", None)
    retrieved_documents = result.get("context", None)
    retrieved_documents_text = [d.page_content for d in retrieved_documents]
    retrieved_documents_metadata = [d.metadata for d in retrieved_documents]
    prompt_template = prompt.template if prompt is not None else None
    embedding_model_name = EMB_MODEL_NAME
    LLM_name = MODEL_NAME if prompt is not None else None

    # Prepare the content to be logged as a dictionary
    log_entry = {
        "user_query": user_query,
        "retrieved_docs_text": retrieved_documents_text,
        "prompt": prompt_template,
        "generated_answer": generated_answer,
        "embedding_model": embedding_model_name,
        "llm": LLM_name,
        "reranker": reranker,
        "retrieved_doc_metadata": retrieved_documents_metadata,
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
    }

    return log_entry
