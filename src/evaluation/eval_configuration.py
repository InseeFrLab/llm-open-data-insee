import copy
from dataclasses import dataclass, field
from typing import Any

from src.config import DefaultFullConfig


@dataclass
class EvalConfiguration:
    name: str = field(default="default_name")
    database: str = field(default="chromadb")
    database_path: str = field(default="")

    def get(self, attribute_name: str, default_value=None) -> Any:
        """
        Get the value of an attribute by its name.
        """
        return getattr(self, attribute_name, default_value)

    def copy(self):
        """
        Create a copy of this configuration.
        """
        return copy.deepcopy(self)


@dataclass
class RetrievalConfiguration(EvalConfiguration):
    # Embedding model
    embedding_model_name: str = field(
        default=DefaultFullConfig().embedding_model, metadata={"description": "embedding model"}
    )
    collection: str | None = field(default=None)
    chunk_size: int | None = field(default=None, metadata={"description": "chunk size"})
    overlap_size: int | None = field(default=None, metadata={"description": "overlap size"})

    # Reranker model
    reranker_type: str | None = field(
        default=None,
        metadata={"description": """Reranker type, choose among: ["BM25", "Cross-encoder", "ColBERT", "Metadata"]"""},
    )
    reranker_name: str | None = field(
        default=None,
        metadata={"description": "Reranker model name (when it exists)"},
    )
    param_ensemble: dict[str, str | None] = field(
        default_factory=dict, metadata={"description": "list of reranker configs"}
    )
    use_metadata: str | None = field(default=None, metadata={"description": "field metadata"})
    rerank_k: int | None = field(default=None, metadata={"description": "field metadata"})

    # Retrieval parameters
    k_values: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30, 40, 50])

    # Parsing metadata
    markdown_separator: list[str] = field(default_factory=lambda: ["\n\n", "\n", ".", " ", ""])
