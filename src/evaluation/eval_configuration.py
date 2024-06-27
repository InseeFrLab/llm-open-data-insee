from abc import ABC
from dataclasses import dataclass, field

from config import COLLECTION_NAME, EMB_MODEL_NAME


@dataclass
class EvalConfiguration(ABC):
    name: str = field(default="default_name")
    database: str = field(default="chromadb")
    database_path: str = field(default="")
    collection: str = field(default=COLLECTION_NAME)


@dataclass
class RetrievalConfiguration(EvalConfiguration):
    embedding_model_name: str = field(default=EMB_MODEL_NAME)
    chunk_size: int = field(default=2000)
    overlap_size: int = field(default=500)
    reranking_method: str = field(default="<to choose>")
    compression_method: str = field(default="<to choose>")
    k_values: list[int] = field(default_factory=lambda: [5, 10, 15])
