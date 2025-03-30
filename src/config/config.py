import logging
import os

import hvac
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


def getenv_from_vault():
    client = hvac.Client(url=os.environ["VAULT_ADDR"], token=os.environ["VAULT_TOKEN"])
    encryptFiles = client.secrets.kv.read_secret_version(
        path="projet-llm-insee-open-data/chatbot", mount_point="onyxia-kv", raise_on_deleted_version=False
    )
    vault_variables = encryptFiles.get("data").get("data")

    return vault_variables


def get_config_s3():
    config_s3 = {
        "endpoint_url": os.getenv("AWS_ENDPOINT_URL", "https://minio.lab.sspcloud.fr"),
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "token": os.getenv("AWS_SESSION_TOKEN"),
    }
    return config_s3


def setenv_from_vault():
    logger.info("Checking environment variables from vault")
    vault_env_vars = getenv_from_vault()
    for key, value in vault_env_vars.items():
        os.environ[key] = value


def get_config_database_qdrant(default_collection_name: str = "dirag_mistral_small"):
    config_database_client = {
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
        "QDRANT_COLLECTION_NAME": os.getenv("COLLECTION_NAME", default_collection_name),
    }
    return config_database_client
