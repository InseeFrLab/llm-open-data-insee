import logging
import os
import subprocess

import s3fs
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %I:%M:%S %p", level=logging.INFO)


def get_file_system(token=None) -> s3fs.S3FileSystem:
    """
    Creates and returns an S3 file system instance using the s3fs library.

    This function configures the S3 file system with endpoint URL and credentials
    obtained from environment variables, enabling interactions with the specified
    S3-compatible storage. Optionally, a security token can be provided for session-based
    authentication.

    Parameters:
    -----------
    token : str, optional
        A temporary security token for session-based authentication. This is optional and
        should be provided when using session-based credentials.

    Returns:
    --------
    s3fs.S3FileSystem
        An instance of the S3 file system configured with the specified endpoint and
        credentials, ready to interact with S3-compatible storage.

    Environment Variables:
    ----------------------
    AWS_S3_ENDPOINT : str
        The S3 endpoint URL for the storage provider, typically in the format `https://{endpoint}`.
    AWS_ACCESS_KEY_ID : str
        The access key ID for authentication.
    AWS_SECRET_ACCESS_KEY : str
        The secret access key for authentication.

    Example:
    --------
    fs = get_file_system(token="your_temporary_token")
    """

    options = {
        "client_kwargs": {"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
    }

    if token is not None:
        options["token"] = token

    return s3fs.S3FileSystem(**options)


def cache_model_from_hf_hub(
    model_name,
    s3_bucket="models-hf",
    s3_cache_dir="hf_hub",
    s3_token=None,
    hf_token=None,
):
    """Use S3 as proxy cache from HF hub if a model is not already cached locally.

    Args:
        model_name (str): Name of the model on the HF hub.
        s3_bucket (str): Name of the S3 bucket to use.
        s3_cache_dir (str): Path of the cache directory on S3.
    """
    assert "MC_HOST_s3" in os.environ, "Please set the MC_HOST_s3 environment variable."

    # Local cache config
    LOCAL_HF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_name_hf_cache = "models--" + "--".join(model_name.split("/"))
    dir_model_local = os.path.join(LOCAL_HF_CACHE_DIR, model_name_hf_cache)

    # Remote cache config
    fs = get_file_system(token=s3_token)
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


if __name__ == "__main__":
    cache_model_from_hf_hub(os.environ["EMB_MODEL_NAME"])
    cache_model_from_hf_hub(os.environ["LLM_MODEL_NAME"])
