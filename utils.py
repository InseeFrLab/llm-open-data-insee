import os
import s3fs
from langchain_core.prompts import PromptTemplate

from src.db_building import load_vector_database
from src.model_building.fetch_llm_model import cache_model_from_hf_hub


def format_docs(docs: list):
    return "\n\n".join(
        [
            f"""
            Doc {i + 1}:\nTitle: {doc.metadata.get("Header 1")}\n
            Source: {doc.metadata.get("url")}\n
            Content:\n{doc.page_content}
            """
            for i, doc in enumerate(docs)
        ]
    )


def create_prompt_from_instructions(
    system_instructions: str, question_instructions: str
):

    template = f"""
    {system_instructions}

    {question_instructions}
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    return custom_rag_prompt


def retrieve_llm_from_cache(model_id):

    cache_model_from_hf_hub(
            model_id,
            s3_bucket=os.environ["S3_BUCKET"],
            s3_cache_dir="models/hf_hub",
            s3_endpoint=f'https://{os.environ["AWS_S3_ENDPOINT"]}',
        )


def retrieve_db_from_cache(
    filesystem: s3fs.S3FileSystem,
    run_id: str = None,
    force: bool = False
):

    db = load_vector_database(
            filesystem=filesystem,
            database_run_id=run_id,
            # hard coded pour le moment
            force=force
        )
    return db
