import sys
import os

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
    TextStreamer,
)
from langchain_huggingface import HuggingFacePipeline

# from src.model_building.custom_hf_pipeline import CustomHuggingFacePipeline
from .fetch_llm_model import cache_model_from_hf_hub

# Add the project root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(""), "./src"))
if root_dir not in sys.path:
    sys.path.append(root_dir)


def build_llm_model(
    model_name,
    quantization_config: bool = False,
    config: bool = False,
    token=None,
    streaming: bool = False,
    generation_args: dict = {},
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

    configs = {
        # Load quantization config
        "quantization_config": (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=False,
            )
            if quantization_config
            else None
        ),
        # Load LLM config
        "config": (
            AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=token)
            if config
            else None
        ),
        "token": token,
    }

    # Load LLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, device_map="auto", token=configs["token"]
    )
    streamer = None
    if streaming:
        streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)

    # Check if tokenizer has a pad_token; if not, set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LLM
    model = AutoModelForCausalLM.from_pretrained(model_name, **configs)

    # Create a pipeline with  tokenizer and model
    pipeline_HF = pipeline(
        task="text-generation",  # TextGenerationPipeline HF pipeline
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=generation_args.get("max_new_tokens", 2000),
        return_full_text=generation_args.get("return_full_text", False),
        device_map="auto",
        do_sample=generation_args.get("do_sample", True),
        temperature=generation_args.get("temperature", None),
        streamer=streamer,
    )
    llm = HuggingFacePipeline(pipeline=pipeline_HF)
    return llm, tokenizer
