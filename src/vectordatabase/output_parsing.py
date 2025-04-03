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
