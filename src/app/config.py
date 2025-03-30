import os

from src.config.config import get_config_database_qdrant, get_config_s3, setenv_from_vault


def create_config_app(use_vault=True):
    setenv_from_vault()

    config_s3 = get_config_s3()

    config_database_client = get_config_database_qdrant(default_collection_name="dirag_mistral_small")

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
