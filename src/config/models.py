import ast
from typing import Any

from confz import BaseConfig
from confz.base_config import BaseConfigMetaclass
from pydantic import validator


class FullConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """
    Full configuration model.

    BaseConfig: allows instantiation using either the usual keyword arguments or through
    a list of configuration sources passed to the constructor's `config_sources` special argument.
    """

    # S3 CONFIG
    aws_s3_endpoint: str
    s3_bucket: str
    s3_endpoint_url: str  # (Templated)

    # LOCAL FILES
    work_dir: str
    relative_data_dir: str
    relative_logs_dir: str
    data_dir_path: str  # (Templated)
    logs_dir_path: str  # (Templated)

    # ML FLOW LOGGING
    experiment_name: str
    mlflow_run_id: str | None = None
    mlflow_tracking_uri: str
    mlflow_load_artifacts: bool

    # RAW DATA PROCESSING
    data_raw_s3_path: str
    raw_dataset_uri: str  # (Templated)
    markdown_split: bool
    separators: list[str]

    rawdata_web4g: str
    rawdata_web4g_uri: str  # (Templated)
    rawdata_rmes: str
    rawdata_rmes_uri: str  # (Templated)

    # PARSING, PROCESSING and CHUNKING
    max_pages: int | None = None
    chunk_size: int
    chunk_overlap: int
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
    batch_size_embedding: int

    # EMBEDDING MODEL
    emb_device: str
    embedding_model: str

    # LLM
    llm_model: str
    quantization: bool
    s3_model_cache_dir: str
    max_new_tokens: int

    # EVALUATION
    faq_s3_path: str
    faq_s3_uri: str  # (Templated)
    topk_stats: int

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

    # Allow the 'None' string to represent None values for all optional parameters
    # (importing from MLFlow requires this)
    @validator("chunk_size", "chunk_overlap", "max_pages", "reranking_method", "retriever_only", pre=True)
    def allow_none(cls, data: Any) -> int | None:
        return None if data == "None" else data

    # AST-parsing of separator list
    @validator("separators", pre=True)
    def json_serialize(cls, data: Any) -> int | None:
        return ast.literal_eval(data) if isinstance(data, str) else data
