import os
import logging

import s3fs
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from ..config import S3_ENDPOINT_URL, S3_BUCKET

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
logger = logging.getLogger(__name__)


def cache_model_from_hf_hub(model_name,
                            s3_bucket=S3_BUCKET,
                            s3_cache_dir="models/hf_hub"):
    """Use S3 as proxy cache from HF hub if a model is not already cached locally.

    Args:
        model_name (str): Name of the model on the HF hub.
        s3_bucket (str): Name of the S3 bucket to use.
        s3_cache_dir (str): Path of the cache directory on S3.
    """
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    LOCAL_HF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_name_hf_cache = 'models--' + '--'.join(model_name.split('/'))
    if model_name_hf_cache not in os.listdir(LOCAL_HF_CACHE_DIR):
        available_models_s3 = [os.path.basename(path) for path in fs.ls(os.path.join(s3_bucket,
                                                                                     s3_cache_dir))]
        dir_model_s3 = os.path.join(s3_bucket, s3_cache_dir, model_name_hf_cache)
        # Try fetching from S3 if available
        if model_name_hf_cache in available_models_s3:
            logger.info(f'Fetching model {model_name} from S3.')
            fs.get(dir_model_s3, LOCAL_HF_CACHE_DIR, recursive=True)
        # Else, fetch from HF Hub and push to S3
        else:
            logger.info(f'Model {model_name} not found on S3, fetching from HF hub.')
            AutoModel.from_pretrained(model_name)
            dir_model_local = os.path.join(LOCAL_HF_CACHE_DIR, model_name_hf_cache)
            logger.info(f'Putting model {model_name} on S3.')
            fs.put(dir_model_local, dir_model_s3, recursive=True)
    else:
        logger.info(f'Model {model_name} found in local cache.')


def build_llm_model(model_name,
                    quantization_config: bool = False,
                    config: bool = False,
                    token=None):
    """
    Create the llm model
    """
    cache_model_from_hf_hub(model_name)

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
        "config": AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=token)
        if config
        else None,
        "token": token,
    }

    # Load LLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, device_map="auto", token=configs["token"]
    )

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
        max_new_tokens=2000,
        temperature=0.2,
        return_full_text=False,
        device_map="auto",
        do_sample=True,
    )

    return HuggingFacePipeline(pipeline=pipeline_HF)
