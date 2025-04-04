import pandas as pd


def langchain_documents_to_df(retrieved_docs):
    """
    Converts a list of retrieved documents into a pandas DataFrame.

    Parameters:
    retrieved_docs (list): A list of documents, where each document has
                           `page_content` and `metadata` attributes.

    Returns:
    pd.DataFrame: A DataFrame containing the page content and metadata of each document.
    """
    result_list = []
    for doc in retrieved_docs:
        row = {"page_content": doc.page_content}
        row.update(doc.metadata)
        result_list.append(row)

    result = pd.DataFrame(result_list)
    return result


def format_docs(docs: list):
    docs_in_prompt = "\n\n".join(
        [
            f"""--------------\n
Doc {i + 1}:\n
Title: {doc.metadata.get("titre", "").replace("#", "")}\n
Abstract: {doc.metadata.get("abstract", "")}\n
Source: {doc.metadata.get("url", "")}\n
Content:\n{doc.page_content or "[No content]"}\n
"""
            for i, doc in enumerate(docs)
        ]
    )
    return docs_in_prompt
