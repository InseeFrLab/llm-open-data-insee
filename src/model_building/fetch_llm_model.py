import logging
import os
import subprocess
from collections.abc import Iterable

import s3fs
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def cache_model_from_hf_hub(
    model_name: str,
    s3_bucket: str = "models-hf",
    s3_cache_dir: str = "hf_hub",
    s3_token: str = None,
    hf_token: str = None,
    config: dict = None,
):
    """Use S3 as proxy cache from HF hub if a model is not already cached locally.

    Args:
        model_name (str): Name of the model on the HF hub.
        s3_bucket (str): Name of the S3 bucket to use.
        s3_cache_dir (str): Path of the cache directory on S3.
    """
    assert "MC_HOST_s3" in os.environ, "Please set the MC_HOST_s3 environment variable."

    if config is None:
        config = {}

    # Local cache config
    LOCAL_HF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_name_hf_cache = "models--" + "--".join(model_name.split("/"))
    dir_model_local = os.path.join(LOCAL_HF_CACHE_DIR, model_name_hf_cache)

    # Remote cache config
    fs = s3fs.S3FileSystem(endpoint_url=config.s3_endpoint_url)
    available_models_s3 = [os.path.basename(path) for path in fs.ls(os.path.join(s3_bucket, s3_cache_dir))]
    dir_model_s3 = os.path.join(s3_bucket, s3_cache_dir, model_name_hf_cache)

    if model_name_hf_cache not in os.listdir(LOCAL_HF_CACHE_DIR):
        # Try fetching from S3 if available
        if model_name_hf_cache in available_models_s3:
            print(f"Fetching model {model_name} from S3.")
            cmd = ["mc", "cp", "-r", f"s3/{dir_model_s3}", f"{LOCAL_HF_CACHE_DIR}/"]
            with open("/dev/null", "w") as devnull:
                subprocess.run(cmd, check=True, stdout=devnull, stderr=devnull)
        # Else, fetch from HF Hub and push to S3
        else:
            print(f"Model {model_name} not found on S3, fetching from HF hub.")
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", token=hf_token)
            print(f"Putting model {model_name} on S3.")
            cmd = [
                "mc",
                "cp",
                "-r",
                f"{dir_model_local}/",
                f"s3/{dir_model_s3}",
            ]
            with open("/dev/null", "w") as devnull:
                subprocess.run(cmd, check=True, stdout=devnull, stderr=devnull)
    else:
        print(f"Model {model_name} found in local cache. ")
        if model_name_hf_cache not in available_models_s3:
            # Push from local HF cache to S3
            print(f"Putting model {model_name} on S3.")
            cmd = [
                "mc",
                "cp",
                "-r",
                f"{dir_model_local}/",
                f"s3/{dir_model_s3}",
            ]
            with open("/dev/null", "w") as devnull:
                subprocess.run(cmd, check=True, stdout=devnull, stderr=devnull)


def cache_models_from_hf_hub(
    models_names: Iterable[str],
    s3_endpoint: str | None = None,
    s3_bucket: str = "models-hf",
    s3_cache_dir: str = "hf_hub",
):
    for model_name in models_names:
        cache_model_from_hf_hub(model_name, s3_endpoint, s3_bucket, s3_cache_dir)
