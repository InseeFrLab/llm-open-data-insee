from dataclasses import dataclass, field
from abc import ABC

from config import EMB_MODEL_NAME, COLLECTION_NAME

@dataclass
class EvalConfiguration(ABC):
    database: str = field(default="chromadb")
    database_path: str = field(default="")
    collection: str = field(default=COLLECTION_NAME)

@dataclass
class RetrievalConfiguration(EvalConfiguration):
    embedding_model: str = field(default=EMB_MODEL_NAME)
    chunk_size: int = field(default=2000)
    overlap_size: int = field(default=500)
    reranking_method: str = field(default="<to choose>")
    chunk_size: int = field(default=10)
