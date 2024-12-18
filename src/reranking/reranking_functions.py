from collections.abc import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever


# Define the compression function
def compress_documents_lambda(
    documents: Sequence[Document], query: str, k: int = 5, **kwargs: dict[str, Any]
) -> Sequence[Document]:
    """Compress retrieved documents given the query context."""

    # Initialize the retriever with the documents
    retriever = BM25Retriever.from_documents(documents, k=k, **kwargs)
    return retriever.get_relevant_documents(query)


def expected_relevance_values(logits, grades_token_ids, list_grades):
    next_token_logits = logits[:, -1, :]
    next_token_logits = next_token_logits.cpu()[0]
    probabilities = F.softmax(next_token_logits[grades_token_ids], dim=-1).numpy()
    return np.dot(np.array(list_grades), probabilities)


def peak_relevance_likelihood(logits, grades_token_ids, list_grades):
    index_max_grade = np.array(list_grades).argmax()
    next_token_logits = logits[:, -1, :]
    probabilities = F.softmax(next_token_logits, dim=-1).cpu().numpy()[0]
    return probabilities[grades_token_ids[index_max_grade]]


def compute_sequence_log_probs(tokenizer, model, sequence):
    # Tokenize the input sequence
    inputs = tokenizer(sequence, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Move input tensors to the same device as the model
    inputs = inputs.to(model.device)

    # Get the logits from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Calculate the probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Compute the probability of the sequence
    sequence_probability = 0.0
    for i in range(len(input_ids[0]) - 1):
        token_id = input_ids[0][i + 1]
        sequence_probability += log_probs[0, i, token_id].item()

    return sequence_probability


def RG_S(tokenizer, model, query, document, aggregating_method, k=5):
    list_grades = list(range(k))
    grades_token_ids = [tokenizer(str(grade))["input_ids"][1] for grade in list_grades]

    RG_S_template = """
    Sur une échelle de 0 à {k}, jugez la pertinence entre la requête et le document.
    Requête : {query}
    Document : {document}
    Réponse : """

    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant chatbot expert en Statistique Publique.",
        },
        {
            "role": "user",
            "content": RG_S_template.format(query=query, document=document, k=k),
        },
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return aggregating_method(logits, grades_token_ids, list_grades)


def RG_4L(tokenizer, model, query, document, args):
    possible_judgements = [
        " Parfaitement Pertinent",
        " Très Pertinent",
        " Assez Pertinent",
        " Non Pertinent",
    ]
    list_grades = np.array([3, 2, 1, 0])
    RG_4L_template = """
    Evaluez la pertinence du document donné par rapport à la question posée.
    Répondez uniquement parmi : Parfaitement Pertinent, Très Pertinent, Assez Pertinent ou Non Pertinent.
    Requête : {query}
    Document : {document}
    Réponse : {judgement}"""

    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant chatbot expert en Statistique Publique.",
        },
        {"role": "user", "content": RG_4L_template},
    ]

    log_probs = []
    for judgement in possible_judgements:
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False).format(
            query=query, document=document, judgement=judgement
        )
        log_probs.append(compute_sequence_log_probs(sequence=input_text))

    probs = F.softmax(torch.tensor(log_probs), dim=-1).numpy()
    return np.dot(probs, list_grades)


def RG_3L(tokenizer, model, query, document, args):
    possible_judgements = [" Très Pertinent", " Assez Pertinent", " Non Pertinent"]
    list_grades = np.array([2, 1, 0])
    RG_3L_template = """
    Evaluez la pertinence du document donné par rapport à la question posée.
    Répondez uniquement parmi : Très Pertinent, Assez Pertinent ou Non Pertinent.
    Requête : {query}
    Document : {document}
    Réponse : {judgement}"""

    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant chatbot expert en Statistique Publique.",
        },
        {"role": "user", "content": RG_3L_template},
    ]

    log_probs = []
    for judgement in possible_judgements:
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False).format(
            query=query, document=document, judgement=judgement
        )
        log_probs.append(compute_sequence_log_probs(sequence=input_text))

    probs = F.softmax(torch.tensor(log_probs), dim=-1).numpy()
    return np.dot(probs, list_grades)


def RG_YN(tokenizer, model, query, document, aggregating_method):
    list_judgements = [" Oui", " Non"]
    grades_token_ids = [tokenizer(j)["input_ids"][1] for j in list_judgements]
    list_grades = [1, 0]

    RG_YN_template = """
    Pour la requête et le document suivants, jugez s'ils sont pertinents. Répondez UNIQUEMENT par Oui ou Non.
    Requête : {query}
    Document : {document}
    Réponse : """

    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant chatbot expert en Statistique Publique.",
        },
        {
            "role": "user",
            "content": RG_YN_template.format(query=query, document=document),
        },
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return aggregating_method(logits, grades_token_ids, list_grades)


def llm_reranking(tokenizer, model, query, retrieved_documents, assessing_method, aggregating_method):
    docs_content = retrieved_documents.copy()  # [doc.page_content for doc in retrieved_documents]

    scores = []
    for document in docs_content:
        score = assessing_method(query, document, aggregating_method)
        scores.append(score)

    docs_with_scores = list(zip(retrieved_documents, scores, strict=False))
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_documents = [doc for doc, score in docs_with_scores]  # docs_with_scores
    return sorted_documents
