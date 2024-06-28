from dataclasses import dataclass, field
from abc import ABC
from typing import List, Dict, Optional

from config import EMB_MODEL_NAME, COLLECTION_NAME


@dataclass
class EvalConfiguration(ABC):
    name: str = field(default="default_name")
    database: str = field(default="chromadb")
    database_path: str = field(default="")
    collection: str = field(default=COLLECTION_NAME)


@dataclass
class RetrievalConfiguration(EvalConfiguration):
    # embedding model
    embedding_model_name: str = field(
        default=EMB_MODEL_NAME, metadata={"description": "embedding model"}
    )
    chunk_size: int = field(default=2000, metadata={"description": "chunck size"})
    overlap_size: int = field(default=500, metadata={"description": "overlap size"})

    # reranker model
    reranker_type: str = field(
        default="<to choose>",
        metadata={
            "description": """Reranker type, choose among : ["BM25","Cross-encoder","ColBERT", "Metadata"]"""
        },
    )
    reranker_name: str = field(
        default="<to choose>",
        metadata={"description": "Reranker model name (when it exists)"},
    )
    param_ensemble: Dict[str, Optional[str]] = field(
        default_factory=dict, metadata={"description": "list of reranker configs"}
    )
    use_metadata: Optional[str] = field(
        default=None, metadata={"description": "field metadata"}
    )
    rerank_k: int = field(default=5, metadata={"description": "field metadata"})

    # retrieval parameters
    k_values: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30, 40, 50])

    # parsing metadata
    markdown_separator: List[str] = field(default_factory=["\n\n", "\n", ".", " ", ""])
