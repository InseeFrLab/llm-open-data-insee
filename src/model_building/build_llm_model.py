from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import torch

from config import MODEL_NAME


def build_llm_model(quantization_config: bool = False, config: bool = False, token=None):
    """
    Create the llm model
    """
    torch.cuda.empty_cache()

    configs = {
        # Load quantization config
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=False,
        )
        if quantization_config
        else None,
        # Load LLM config
        "config": AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True, token=token)
        if config
        else None,
        "token": token,
    }

    # Load LLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=True, device_map="auto", token=configs["token"]
    )

    # Check if tokenizer has a pad_token; if not, set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **configs)

    # Create a pipeline with  tokenizer and model
    pipeline_HF = pipeline(
        task="text-generation",  # TextGenerationPipeline HF pipeline
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2000,
        temperature=0.2,
        return_full_text=False,
        device_map="auto",
        do_sample=True,
    )

    return HuggingFacePipeline(pipeline=pipeline_HF)
