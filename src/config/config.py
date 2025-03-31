import os
from loguru import logger

import hvac
from dotenv import load_dotenv

load_dotenv(override=True)

# import logging
# logger = logging.getLogger(__name__)


def getenv_from_vault():
    client = hvac.Client(url=os.environ["VAULT_ADDR"], token=os.environ["VAULT_TOKEN"])
    encryptFiles = client.secrets.kv.read_secret_version(
        path="projet-llm-insee-open-data/chatbot", mount_point="onyxia-kv", raise_on_deleted_version=False
    )
    vault_variables = encryptFiles.get("data").get("data")

    return vault_variables


def get_config_s3(**kwargs):
    if "verbose" in kwargs and kwargs["verbose"] is True:
        logger.info("Setting 'endpoint_url', 'key', 'secret' and 'token' parameters")

    config_s3 = {
        "endpoint_url": os.getenv("AWS_ENDPOINT_URL", "https://minio.lab.sspcloud.fr"),
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "token": os.getenv("AWS_SESSION_TOKEN"),
    }
    return config_s3


def setenv_from_vault():
    logger.info("Setting environment variables from values provided to Vault")
    vault_env_vars = getenv_from_vault()
    for key, value in vault_env_vars.items():
        os.environ[key] = value


def get_config_database_qdrant(default_collection_name: str = "dirag_mistral_small", **kwargs):
    if "verbose" in kwargs and kwargs["verbose"] is True:
        logger.info("Setting 'QDRANT_URL', 'QDRANT_API_KEY' and 'QDRANT_COLLECTION_NAME' parameters")

    config_database_client = {
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
        "QDRANT_COLLECTION_NAME": os.getenv("COLLECTION_NAME", default_collection_name),
    }
    return config_database_client


def get_config_mlflow(mlflow_experiment_name: str = "experiment_name", **kwargs):
    if "verbose" in kwargs and kwargs["verbose"] is True:
        logger.info("Setting 'MLFLOW_TRACKING_URI' and 'MLFLOW_EXPERIMENT_NAME' parameters")

    config_mlflow = {
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", None),
        "MLFLOW_EXPERIMENT_NAME": mlflow_experiment_name,
    }
    return config_mlflow


def get_config_openai(suffix: str = "", default_values: dict = None, **kwargs):
    if suffix.startswith("_") is False and suffix != "":
        suffix = "_" + suffix

    if default_values is None:
        default_values = {}

    if "verbose" in kwargs and kwargs["verbose"] is True:
        logger.info(f"Setting 'OPENAI_API_BASE{suffix}' and 'OPENAI_API_KEY{suffix}' parameters")

    config_openai = {
        f"OPENAI_API_BASE{suffix}": default_values.get("OPENAI_API_BASE", os.getenv("OPENAI_API_BASE")),
        f"OPENAI_API_KEY{suffix}": default_values.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
    }

    return config_openai


def get_config_vllm(url_embedding_model: str = None, url_generative_model: str = None, **kwargs):
    if url_embedding_model is not None:
        config = get_config_openai(
            suffix="EMBEDDING", default_values={"OPENAI_API_BASE": url_embedding_model}, **kwargs
        )
    else:
        config = {}

    if url_generative_model is not None:
        config_generative = get_config_openai(
            suffix="GENERATIVE", default_values={"OPENAI_API_BASE": url_generative_model}, **kwargs
        )
        config = {**config, **config_generative}

    return config


def set_config(
    use_vault: bool = False,
    components: list = None,
    database_manager: str = "Qdrant",
    models_location: dict = None,
    override: dict = None,
    verbose: bool = False,
    **kwargs,
):
    if database_manager.lower() != "qdrant":
        message = (
            "Only Qdrant database is handled properly right now."
            "If you want to use another provider (Milvus, Chroma...), "
            "you are on your own"
        )
        raise ValueError(message)

    if models_location is None:
        models_location = {}

    kwargs = {**kwargs, **{"verbose": verbose}}

    if use_vault is True:
        setenv_from_vault()

    config = {}

    if components is None:
        return config

    if "s3" in components:
        config = {**config, **get_config_s3(**kwargs)}

    if "mlflow" in components:
        config = {**config, **get_config_mlflow(**kwargs)}

    if "database" in components and database_manager.lower() == "qdrant":
        config = {**config, **get_config_database_qdrant(**kwargs)}

    if "model" in components:
        config = {**config, **get_config_vllm(**models_location, **kwargs)}

    if override is not None:
        for key, value in override.items():
            config[key] = value

    return config
