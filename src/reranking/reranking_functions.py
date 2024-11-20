from collections.abc import Any, Sequence

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever


# Define the compression function
def compress_documents_lambda(documents: Sequence[Document], query: str, k: int = 5, **kwargs: dict[str, Any]) -> Sequence[Document]:
    """Compress retrieved documents given the query context."""

    # Initialize the retriever with the documents
    retriever = BM25Retriever.from_documents(documents, k=k, **kwargs)
    return retriever.get_relevant_documents(query)
