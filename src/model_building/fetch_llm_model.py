import logging
import os
import subprocess
from collections.abc import Iterable

import s3fs
from transformers import AutoModelForCausalLM

from src.config import default_config, load_config, models_only_argparser

logger = logging.getLogger(__name__)


def get_file_system() -> s3fs.S3FileSystem:
    """
    Return the s3 file system.
    """
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def cache_model_from_hf_hub(
    model_name: str,
    s3_endpoint: str | None = None,
    s3_bucket: str = "models-hf",
    s3_cache_dir: str = "hf_hub",
):
    """Use S3 as proxy cache from HF hub if a model is not already cached locally.

    Args:
        model_name (str, optional): Name of the model on the HF hub.
        s3_bucket (str): Name of the S3 bucket to use.
        s3_cache_dir (str, optional): Path of the cache directory on S3.
    """
    assert "MC_host_s3" in os.environ, "Please set the MC_host_s3 environment variable."

    if s3_bucket is None:
        s3_bucket = os.environ["S3_BUCKET"]
    if s3_endpoint is None:
        s3_endpoint = f'https://{os.environ["AWS_S3_ENDPOINT"]}'
    # Local cache config
    LOCAL_HOME = os.path.expanduser("~")
    LOCAL_HF_CACHE_DIR = os.path.join(LOCAL_HOME, ".cache", "huggingface", "hub")
    model_name_hf_cache = "models--" + model_name.replace("/", "--")
    dir_model_local = os.path.join(LOCAL_HF_CACHE_DIR, model_name_hf_cache)

    # Remote cache config
    fs = s3fs.S3FileSystem(endpoint_url=s3_endpoint)
    available_models_s3 = [os.path.basename(path) for path in fs.ls(os.path.join(s3_bucket, s3_cache_dir))]
    dir_model_s3 = os.path.join(s3_bucket, s3_cache_dir, model_name_hf_cache)

    if model_name_hf_cache not in os.listdir(LOCAL_HF_CACHE_DIR):
        # Try fetching from S3 if available
        if model_name_hf_cache in available_models_s3:
            print(f"Fetching model {model_name} from S3.")
            cmd = [
                "mc",
                "cp",
                "-r",
                f"s3/{dir_model_s3}",
                f"{LOCAL_HF_CACHE_DIR}/",
                "> dev/null",
            ]
            subprocess.run(cmd, check=True)
        # Else, fetch from HF Hub and push to S3
        else:
            print(f"Model {model_name} not found on S3, fetching from HF hub.")
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
            )
            print(f"Putting model {model_name} on S3.")
            cmd = [
                "mc",
                "cp",
                "-r",
                f"{dir_model_local}/",
                f"s3/{dir_model_s3}",
                "> dev/null",
            ]
            subprocess.run(cmd, check=True)
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
                "> dev/null",
            ]
            subprocess.run(cmd, check=True)


def cache_models_from_hf_hub(
    models_names: Iterable[str],
    s3_endpoint: str | None = None,
    s3_bucket: str = "models-hf",
    s3_cache_dir: str = "hf_hub",
):
    for model_name in models_names:
        cache_model_from_hf_hub(model_name, s3_endpoint, s3_bucket, s3_cache_dir)


if __name__ == "__main__":
    load_config(models_only_argparser())
    cache_models_from_hf_hub(
        [default_config["emb_model"], default_config["llm_model"]],
        default_config.get("s3_bucket"),
        default_config.get("s3_endpoint_url"),
        default_config.get("s3_model_cache_dir"),
    )
