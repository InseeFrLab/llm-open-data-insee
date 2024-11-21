import logging
import os
import subprocess
from collections.abc import Iterable

import s3fs
from transformers import AutoModelForCausalLM

from src.config import Configurable, DefaultFullConfig, FullConfig, models_only_argparser, process_args

logger = logging.getLogger(__name__)


@Configurable()
def get_s3_file_system(config: FullConfig = DefaultFullConfig()) -> s3fs.S3FileSystem:
    """
    Return the s3 file system.
    """
    return s3fs.S3FileSystem(endpoint_url=config["s3_endpoint_url"])


@Configurable()
def cache_model_from_hf_hub(
    model_name: str,
    s3_bucket: str = "models-hf",
    s3_cache_dir: str = "hf_hub",
    config: FullConfig = DefaultFullConfig(),
):
    """Use S3 as proxy cache from HF hub if a model is not already cached locally.

    Args:
        model_name (str, optional): Name of the model on the HF hub.
        s3_bucket: Name of the S3 bucket to use.
        s3_cache_dir (str, optional): Path of the cache directory on S3.
        s3_endpoint_url (str): Overrides config value.
        config: configuration object.
    """
    # Local cache config
    LOCAL_HOME = os.path.expanduser("~")
    LOCAL_HF_CACHE_DIR = os.path.join(LOCAL_HOME, ".cache", "huggingface", "hub")
    model_name_hf_cache = "models--" + model_name.replace("/", "--")
    dir_model_local = os.path.join(LOCAL_HF_CACHE_DIR, model_name_hf_cache)

    # Remote cache config
    fs = s3fs.S3FileSystem(endpoint_url=config.s3_endpoint_url)
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
    process_args(models_only_argparser())
    config = DefaultFullConfig()
    cache_models_from_hf_hub(
        [config.emb_model, config.llm_model],
        config.s3_bucket,
        config.s3_endpoint_url,
        config.s3_model_cache_dir,
    )
