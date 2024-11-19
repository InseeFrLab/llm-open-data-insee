import ast
import os
from dataclasses import dataclass
from typing import Any

import mlflow
import toml
from confz import BaseConfig, CLArgSource, ConfigSource, DataSource, EnvSource, FileSource
from confz.base_config import BaseConfigMetaclass
from confz.loaders import Loader, register_loader
from pydantic import validator

default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag.toml")


@dataclass
class MLFlowSource(ConfigSource):
    pass


class MLFlowLoader(Loader):
    @classmethod
    def populate_config(cls, config: dict, config_source: MLFlowSource):
        if config.get("mlflow_run_id") and config.get("mlflow_tracking_uri"):
            client = mlflow.tracking.MlflowClient(tracking_uri=config["mlflow_tracking_uri"])
            mlflow_params = client.get_run(config["mlflow_run_id"]).data.params
            # Do not override ML Flow loading parameters
            mlflow_params.pop("experiment_name", None)
            mlflow_params.pop("mlflow_tracking_uri", None)
            mlflow_params.pop("mlflow_run_id", None)
            cls.update_dict_recursively(config, mlflow_params)


register_loader(MLFlowSource, MLFlowLoader)


@dataclass
class TemplatePassSource(ConfigSource):
    pass


class TemplatePassLoader(Loader):
    @classmethod
    def populate_config(cls, config: dict, config_source: TemplatePassSource):
        templated_params = config.get("__templated_params__")
        if templated_params:
            for p in templated_params:
                if config.get(p):
                    print(config)
                    cls.update_dict_recursively(config, {p: config[p].format(**config)})


register_loader(TemplatePassSource, TemplatePassLoader)


class BaseRAGConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    # S3 CONFIG
    experiment_name: str
    aws_s3_endpoint: str
    s3_bucket: str
    s3_endpoint_url: str  # (Templated)

    # LOCAL FILES
    work_dir: str
    relative_data_dir: str
    log_file_path: str  # (Templated)
    relative_log_dir: str

    # ML FLOW LOGGING
    mlflow_run_id: str | None = None
    mlflow_tracking_uri: str
    mlflow_load_artifacts: bool

    # RAW DATA PROCESSING
    data_raw_s3_path: str
    raw_dataset_uri: str  # (Templated)
    markdown_split: bool
    use_tokenizer_to_chunk: bool
    separators: list[str]

    rawdata_web4g: str
    rawdata_web4g_uri: str  # (Templated)
    rawdata_rmes: str
    rawdata_rmes_uri: str  # (Templated)

    # PARSING, PROCESSING and CHUNKING
    max_pages: int | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    documents_s3_dir: str  # (Templated)
    documents_jsonl_s3_path: str  # (Templated)
    documents_parquet_s3_path: str  # (Templated)

    # VECTOR DATABASE
    chroma_db_local_dir: str
    chroma_db_local_path: str  # (Templated)
    chroma_db_s3_dir: str
    collection_name: str
    force_rebuild: bool
    chroma_db_s3_path: str  # (Templated)

    # EMBEDDING MODEL
    emb_device: str
    emb_model: str

    # LLM
    llm_model: str
    quantization: bool
    s3_model_cache_dir: str
    max_new_tokens: int

    # EVALUATION
    faq_s3_path: str
    faq_s3_uri: str  # (Templated)

    # INSTRUCTION PROMPT
    BASIC_RAG_PROMPT_TEMPLATE: str
    RAG_PROMPT_TEMPLATE: str

    # CHATBOT TEMPLATE
    CHATBOT_SYSTEM_INSTRUCTION: str

    # DATA
    RAW_DATA: str
    LS_DATA_PATH: str
    LS_ANNOTATIONS_PATH: str

    # CHAINLIT
    uvicorn_timeout_keep_alive: int
    cli_message_separator_length: int
    llm_temperature: float
    return_full_text: bool
    do_sample: bool
    temperature: float
    rep_penalty: float
    top_p: float
    reranking_method: str | None = None
    retriever_only: bool | None = None

    # Allow the 'None' string to represent None values for all optional parameters (MLFlow import requires this)
    @validator("chunk_size", "chunk_overlap", "max_pages", "reranking_method", "retriever_only", pre=True)
    def allow_none(cls, data: Any) -> int | None:
        return None if data == "None" else data

    # AST-parsing of separator list
    @validator("separators", pre=True)
    def json_serialize(cls, data: Any) -> int | None:
        return ast.literal_eval(data) if isinstance(data, str) else data


# Singleton mechanism:
# - RAGConfig cannot be initialised with custom keyword arguments
# - Calls to the constructor RAGConfig() are cached (and basically free), the config is not reloaded
class RAGConfig(BaseRAGConfig, metaclass=BaseConfigMetaclass):
    CONFIG_SOURCES = [
        # Set default parameters from default config file
        FileSource(file=default_config_path),
        # Set parameters from config file from env
        FileSource(file_from_env="RAG_CONFIG_FILE", optional=True),
        # Set parameter xxxx directly with the XXXX env variable
        EnvSource(allow=["AWS_S3_ENDPOINT", "WORK_DIR"]),
        # Set parameter xxxx using the RAG_XXXX (case insensitive) env variable
        EnvSource(allow_all=True, prefix="RAG_"),
        # Set parameters from config file from command line argument
        FileSource(file_from_cl="--config_file", optional=True),
        # Set parameters from command line argument
        CLArgSource(
            remap={
                # Add explicit command line arguments remapping if needed
                "config_mlflow": "mlflow_run_id"
            }
        ),
        # Set parameters from a previous MLFlow run identified with its mlflow_run_id
        MLFlowSource(),
        # Final pass to template all parameters listed in __templated_params__
        TemplatePassSource(),
    ]


def custom_config(defaults: dict | None = None, overrides: dict | None = None):
    """
    Load a configuration from files, environment and command line argument but:
    - Default values are taken from [defaults] (if specified) rather than from the default file
    - All values are overriden with [overrides] (if specified)
    """
    defaults = {k.lower(): v for k, v in defaults.items()} if defaults else {}
    overrides = {k.lower(): v for k, v in overrides.items()} if overrides else {}
    return BaseRAGConfig(
        config_sources=[
            FileSource(file=default_config_path),  # Load defaults
            DataSource(data=defaults),  # Override default with custom defaults
        ]
        + RAGConfig.CONFIG_SOURCES[1:-1]  # Load all other sources
        + [
            DataSource(data=overrides),  # Override with custom overrides
            TemplatePassSource(),  # Final templating pass
        ]
    )


if __name__ == "__main__":
    print(toml.dumps(vars(RAGConfig())))
