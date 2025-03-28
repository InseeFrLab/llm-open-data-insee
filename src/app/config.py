import os

import hvac
from loguru import logger


def getenv_from_vault():
    client = hvac.Client(url=os.environ["VAULT_ADDR"], token=os.environ["VAULT_TOKEN"])
    encryptFiles = client.secrets.kv.read_secret_version(
        path="projet-llm-insee-open-data/chatbot", mount_point="onyxia-kv", raise_on_deleted_version=False
    )
    vault_variables = encryptFiles.get("data").get("data")

    return vault_variables


def create_config_app(use_vault=True):
    if use_vault is True:
        logger.info("Checking environment variables from vault")
        vault_env_vars = getenv_from_vault()
        for key, value in vault_env_vars.items():
            os.environ[key] = value

    config_s3 = {
        "AWS_ENDPOINT_URL": os.getenv("AWS_ENDPOINT_URL", "https://minio.lab.sspcloud.fr"),
    }

    config_database_client = {
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
        "QDRANT_COLLECTION_NAME": os.getenv("COLLECTION_NAME", "dirag_mistral_small"),
    }

    config_embedding_model = {
        "OPENAI_API_BASE_EMBEDDING": os.getenv("OPENAI_API_BASE", os.getenv("URL_EMBEDDING_MODEL")),
        "OPENAI_API_KEY_EMBEDDING": os.getenv("OPENAI_API_KEY", "EMPTY"),
    }

    config_generative_model = {
        "OPENAI_API_BASE_GENERATIVE": os.getenv("OPENAI_API_BASE", os.getenv("URL_GENERATIVE_MODEL")),
        "OPENAI_API_KEY_GENERATIVE": os.getenv("OPENAI_API_KEY", "EMPTY"),
    }

    config = {**config_s3, **config_database_client, **config_embedding_model, **config_generative_model}

    return config
