def format_docs(docs) -> str:
    """Convert the retrieved documents to a string to be inserted into the prompt"""
    return "\n\n".join(doc.page_content for doc in docs)
