import os

from langchain_community.llms import VLLM
from transformers import AutoTokenizer

from .fetch_llm_model import cache_model_from_hf_hub


def build_llm_model(
    model_name,
    token: str = None
):
    """
    Create the llm model
    """
    cache_model_from_hf_hub(
        model_name,
        s3_bucket="projet-llm-insee-open-data",
        s3_cache_dir="models/hf_hub",
        s3_endpoint=f'https://{os.environ["AWS_S3_ENDPOINT"]}',
    )

    # Load LLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=True,
                                              device_map="auto",
                                              token=token
                                              )

    # Check if tokenizer has a pad_token; if not, set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = VLLM(
        model=model_name,
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=2000,
        top_k=10,
        top_p=0.95,
        temperature=0.2,
        gpu_memory_utilization=0.8
    )

    return llm
