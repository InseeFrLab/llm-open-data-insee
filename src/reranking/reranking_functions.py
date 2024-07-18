from collections.abc import Sequence
from typing import Any

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever


# Define the compression function
def compress_documents_lambda(documents: Sequence[Document], query: str, k: int = 5, **kwargs: dict[str, Any]) -> Sequence[Document]:
    """Compress retrieved documents given the query context."""

    # Initialize the retriever with the documents
    retriever = BM25Retriever.from_documents(documents, k=k, **kwargs)
    return retriever.get_relevant_documents(query)

RG_YN = """
Pour la requête et le document suivants, jugez s'ils sont pertinents. Répondez par "Oui" ou "Non".
Requête : {query}
Document : {document}
Réponse : """

RG_2L = """
Pour la requête et le document suivants, jugez s'ils sont pertinents. Répondez par "Oui" ou "Non".
Requête : {query}
Document : {document}
Réponse : """

RG_3L = """
Pour la requête et le document suivants, jugez s'ils sont "Très Pertinents", "Assez Pertinents" ou "Non Pertinents".
Requête : {query}
Document : {document}
Réponse : """

RG_4L = """
Pour la requête et le document suivants, jugez s'ils sont "Parfaitement Pertinents", "Très Pertinents", "Assez Pertinents" ou "Non Pertinents".
Requête : {query}
Document : {document}
Réponse : """

RG_S = """
Sur une échelle de 0 à {k}, jugez la pertinence entre la requête et le document.
Requête : {query}
Document : {document}
Réponse : """ 

RELEVANCE_TEMPLATE = [
    {
        "role": "user",
        "content": RG_S,
    }
]

def parse_generated_answer(answer: str) -> bool:
    """
    Parse the generated answer from the LLM to determine relevance.
    Assumes the answer starts with "Yes" or "No".
    """
    ans = answer.lower().strip().split()
    return "yes" in ans

def rerank_LLM(pipe, tokenizer, documents: Sequence[Document], query: str, k: int = 5, **kwargs: dict[str, Any]) -> Sequence[Document]:

    template = tokenizer.apply_chat_template(RELEVANCE_TEMPLATE, tokenize=False, add_generation_prompt=True)
    batch_prompts = [ template.format(query=query, document=doc.page_content, k=4) for doc in documents]
    generated_answers = pipe.invoke(batch_prompts)
    bool_list = [parse_generated_answer(ans) for ans in generated_answers]
    # Filter the documents based on the relevance boolean list
    relevant_documents = [doc for doc, is_relevant in zip(documents, bool_list) if is_relevant]
    top_relevant_documents = relevant_documents[:k]
    return top_relevant_documents


