import ast
import os
from dataclasses import dataclass
from typing import Any

import mlflow
import toml
from confz import BaseConfig, CLArgSource, ConfigSource, EnvSource, FileSource
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
            mlflow_params.pop("experiment_name", None)
            mlflow_params.pop("mlflow_tracking_uri", None)
            mlflow_params.pop("mlflow_run_id", None)
            cls.update_dict_recursively(config, mlflow_params)
        else:
            print("No loading from MLFLOW")
            print(config)


register_loader(MLFlowSource, MLFlowLoader)


class RAGConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    # S3 CONFIG
    experiment_name: str
    aws_s3_endpoint: str
    s3_bucket: str

    # LOCAL FILES
    relative_data_dir: str
    relative_log_dir: str

    # ML FLOW LOGGING
    mlflow_tracking_uri: str
    mlflow_load_artifacts: bool

    # RAW DATA PROCESSING
    data_raw_s3_path: str
    markdown_split: bool
    use_tokenizer_to_chunk: bool
    separators: list[str]

    rawdata_web4g: str
    rawdata_rmes: str

    # PARSING, PROCESSING and CHUNKING
    max_pages: int | None = None
    chunk_size: int | None
    chunk_overlap: int | None

    # VECTOR DATABASE
    chroma_db_local_dir: str
    chroma_db_s3_dir: str
    collection_name: str
    force_rebuild: bool

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

    # Optional
    @validator("chunk_size", "chunk_overlap", "max_pages", pre=True)
    def allow_none(cls, data: Any) -> int | None:
        return None if data == "None" else data

    # AST-parsing of separator list
    @validator("separators", pre=True)
    def json_serialize(cls, data: Any) -> int | None:
        return ast.literal_eval(data) if isinstance(data, str) else data

    CONFIG_SOURCES = [
        FileSource(file=default_config_path),
        FileSource(file_from_env="RAG_CONFIG_FILE", optional=True),
        EnvSource(
            allow_all=True,
            prefix="RAG_",
            remap={
                # Add explicit env variable remapping if needed
            },
        ),
        FileSource(file_from_cl="--config_file", optional=True),
        CLArgSource(
            remap={
                # Add explicit command line arguments remapping if needed
                "config_mlflow": "mlflow_run_id"
            }
        ),
        MLFlowSource(),
    ]


if __name__ == "__main__":
    print(toml.dumps(vars(RAGConfig())))
