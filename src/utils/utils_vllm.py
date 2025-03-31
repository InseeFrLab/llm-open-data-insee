import logging
import os

import requests
from langchain_openai import OpenAIEmbeddings

# Configure logger
logger = logging.getLogger(__name__)


def get_model_from_env(env_var_api: str = "URL_EMBEDDING_MODEL", config_dict: dict = None) -> str:

    url_model = config_dict.get(env_var_api) if config_dict is not None else os.getenv(env_var_api)

    if url_model:
        logger.debug(
            f"Model called from {url_model} API (inferred from {env_var_api} environment variable)"
        )
        try:
            response = requests.get(f"{url_model}models")
            response.raise_for_status()  # Ensure the request was successful
            available_models = response.json().get("data", [])

            if available_models and isinstance(available_models, list):
                return available_models[0].get("id", "")

            logger.warning("No models found in API response.")
        except requests.RequestException as e:
            logger.error(f"Error fetching model from API: {e}")

    local_env_var = env_var_api.replace("URL_", "")
    logger.debug(f"Using local model, checking if value provided in {local_env_var}")

    return os.getenv(local_env_var, "OrdalieTech/Solon-embeddings-large-0.1")


def get_model_max_len(
    model_id: str = "mistralai/Mistral-Small-24B-Instruct-2501",
    env_var_api: str = "URL_EMBEDDING_MODEL",
) -> str:
    url_model = os.getenv(env_var_api)
    response = requests.get(f"{url_model}models")
    response.raise_for_status()  # Ensure the request was successful
    available_models = response.json().get("data", [])
    max_model_len = [model["max_model_len"] for model in available_models if model["id"] == model_id][0]
    return max_model_len


def _embedding_client_api(config, embedding_model):
    emb_model = OpenAIEmbeddings(
        model=embedding_model,
        base_url=config.get("OPENAI_API_BASE_EMBEDDING"),
        api_key=config.get("OPENAI_API_KEY_EMBEDDING"),
    )

    return emb_model
