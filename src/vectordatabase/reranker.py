import requests


def rerank_documents(url, model, query, documents):

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    
    payload = {
        "model": model,
        "query": query,
        "documents": documents
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")


def flatten_results(response_json):
    results = response_json.get("results", [])
    flat_data = []

    for result in results:
        flat_data.append({
            "index": result["index"],
            "text": result["document"]["text"],
            "relevance_score": result["relevance_score"]
        })

    return flat_data


def rerank_top_documents(query, docs_retrieved, url_reranker, model_reranker, top_k=5):
    """
    Rerank documents based on a query using a remote reranker service.

    Parameters:
    - query: str, the query string
    - docs_retrieved: list of LangChain Document objects (or similar with .page_content and .metadata)
    - url_reranker: str, URL of the reranker service
    - model_reranker: str, model name for reranking
    - top_k: int, number of top results to return

    Returns:
    - List of top_k documents with relevance scores added to metadata
    """
    
    # For reranking we must use root rather than /v1 endpoint
    url_reranker = url_reranker.replace("v1/", "rerank/")

    documents = [doc.page_content for doc in docs_retrieved]
    result = rerank_documents(
        url=url_reranker, model=model_reranker, query=query, documents=documents
    )

    docs_reranked = flatten_results(result)[:top_k]
    index_to_score = {
        doc["index"]: doc["relevance_score"] for doc in docs_reranked
    }

    selected_docs = []
    for i, doc in enumerate(docs_retrieved):
        if i in index_to_score:
            doc.metadata["relevance_score"] = index_to_score[i]
            selected_docs.append(doc)

    return selected_docs


# custom class to invoke reranker
class RerankerRetriever:
    def __init__(self, retriever, url_reranker, model_reranker, top_k=5):
        """
        Wraps a retriever with reranking capability.

        Parameters:
        - retriever: a base retriever object with an .invoke(query) method
        - url_reranker: URL of the reranker service
        - model_reranker: model identifier for the reranker
        - top_k: number of top reranked documents to return
        """
        self.retriever = retriever
        self.url_reranker = url_reranker
        self.model_reranker = model_reranker
        self.top_k = top_k

    def invoke(self, query):
        """
        Retrieves and reranks documents for a given query.

        Parameters:
        - query: str, the search query

        Returns:
        - List of top reranked documents
        """
        docs_retrieved = self.retriever.invoke(query)
        return rerank_top_documents(
            query=query,
            docs_retrieved=docs_retrieved,
            url_reranker=self.url_reranker,
            model_reranker=self.model_reranker,
            top_k=self.top_k
        )



