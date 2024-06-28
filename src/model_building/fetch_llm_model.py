import logging
import os

import s3fs
from transformers import AutoModel


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %I:%M:%S %p", level=logging.INFO
)


def cache_model_from_hf_hub(
    model_name,
    s3_endpoint=f'https://{os.environ["AWS_S3_ENDPOINT"]}',
    s3_bucket=os.environ["S3_BUCKET"],
    s3_cache_dir="models/hf_hub",
):
    """Use S3 as proxy cache from HF hub if a model is not already cached locally.

    Args:
        model_name (str): Name of the model on the HF hub.
        s3_bucket (str): Name of the S3 bucket to use.
        s3_cache_dir (str): Path of the cache directory on S3.
    """
    # Local cache config
    LOCAL_HF_CACHE_DIR = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub"
    )
    model_name_hf_cache = "models--" + "--".join(model_name.split("/"))
    dir_model_local = os.path.join(LOCAL_HF_CACHE_DIR, model_name_hf_cache)

    # Remote cache config
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": s3_endpoint})
    available_models_s3 = [
        os.path.basename(path) for path in fs.ls(os.path.join(s3_bucket, s3_cache_dir))
    ]
    dir_model_s3 = os.path.join(s3_bucket, s3_cache_dir, model_name_hf_cache)

    if model_name_hf_cache not in os.listdir(LOCAL_HF_CACHE_DIR):
        # Try fetching from S3 if available
        if model_name_hf_cache in available_models_s3:
            logger.info(f"Fetching model {model_name} from S3.")
            fs.get(dir_model_s3, LOCAL_HF_CACHE_DIR, recursive=True)
        # Else, fetch from HF Hub and push to S3
        else:
            logger.info(f"Model {model_name} not found on S3, fetching from HF hub.")
            AutoModel.from_pretrained(model_name)
            logger.info(f"Putting model {model_name} on S3.")
            fs.put(dir_model_local, dir_model_s3, recursive=True)
    else:
        logger.info(f"Model {model_name} found in local cache. ")
        if model_name_hf_cache not in available_models_s3:
            # Push from local HF cache to S3
            logger.info(f"Putting model {model_name} on S3.")
            fs.put(dir_model_local, dir_model_s3, recursive=True)


if __name__ == "__main__":
    cache_model_from_hf_hub(os.environ["EMB_MODEL_NAME"])
    cache_model_from_hf_hub(os.environ["LLM_MODEL_NAME"])
