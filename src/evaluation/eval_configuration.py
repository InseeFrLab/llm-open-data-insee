from dataclasses import dataclass, field
from abc import ABC
from typing import List, Dict, Optional, Any

from config import EMB_MODEL_NAME, COLLECTION_NAME

@dataclass
class EvalConfiguration(ABC):
    name: str = field(default="default_name")
    database: str = field(default="chromadb")
    database_path: str = field(default="")
    collection: str = field(default=COLLECTION_NAME)

    def get(self, attribute_name: str, default_value=None) -> Any:
        """
        Get the value of an attribute by its name.
        """
        return getattr(self, attribute_name, default_value)


@dataclass
class RetrievalConfiguration(EvalConfiguration):
    # Embedding model
    embedding_model_name: str = field(
        default=EMB_MODEL_NAME, metadata={"description": "embedding model"}
    )
    chunk_size: int = field(default=2000, metadata={"description": "chunk size"})
    overlap_size: int = field(default=500, metadata={"description": "overlap size"})

    # Reranker model
    reranker_type: str = field(
        default=None,
        metadata={
            "description": """Reranker type, choose among: ["BM25", "Cross-encoder", "ColBERT", "Metadata"]"""
        },
    )
    reranker_name: str = field(
        default=None,
        metadata={"description": "Reranker model name (when it exists)"},
    )
    param_ensemble: Dict[str, Optional[str]] = field(
        default_factory=dict, metadata={"description": "list of reranker configs"}
    )
    use_metadata: Optional[str] = field(
        default=None, metadata={"description": "field metadata"}
    )
    rerank_k: int = field(default=5, metadata={"description": "field metadata"})

    # Retrieval parameters
    k_values: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30, 40, 50])

    # Parsing metadata
    markdown_separator: List[str] = field(default_factory=lambda: ["\n\n", "\n", ".", " ", ""])
