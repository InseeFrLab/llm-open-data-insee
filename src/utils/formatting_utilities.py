def format_docs(docs):
    return "\n\n".join([f"Doc{i}:\nTitle: {doc.metadata["title"]}\nContent:\n{doc.page_content}" for i, doc in enumerate(docs)])
