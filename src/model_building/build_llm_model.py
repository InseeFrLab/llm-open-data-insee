import os
import sys
from typing import Any

from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
    pipeline,
)

from src.config import Configurable, DefaultFullConfig, FullConfig

# from src.model_building.custom_hf_pipeline import CustomHuggingFacePipeline
from .fetch_llm_model import cache_model_from_hf_hub

# Add the project root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(""), "./src"))
if root_dir not in sys.path:
    sys.path.append(root_dir)


@Configurable()
def build_llm_model(
    model_name: str,
    load_LLM_config: bool = False,
    streaming: bool = False,
    hf_token: str | None = None,
    config: FullConfig = DefaultFullConfig(),
) -> tuple[HuggingFacePipeline, Any]:
    """
    Create the LLM model
    """
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    cache_model_from_hf_hub(
        model_name,
        s3_bucket=config.s3_bucket,
        s3_cache_dir=config.s3_model_cache_dir,
        s3_endpoint=config.s3_endpoint_url,
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
            if config.quantization
            else None
        ),
        # Load LLM config
        "config": (
            AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=hf_token) if load_LLM_config else None
        ),
        "token": hf_token,
    }

    # Load LLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, device_map="auto", token=hf_token)
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
        max_new_tokens=config.max_new_tokens,
        return_full_text=config.return_full_text,
        device_map="auto",
        do_sample=config.do_sample,
        temperature=config.temperature,
        streamer=streamer,
    )
    llm = HuggingFacePipeline(pipeline=pipeline_HF)
    return llm, tokenizer
