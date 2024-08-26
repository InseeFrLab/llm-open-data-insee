from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever


# Define the compression function
def compress_documents_lambda(documents: Sequence[Document], query: str, k: int = 5, **kwargs: dict[str, Any]) -> Sequence[Document]:
    """Compress retrieved documents given the query context."""

    # Initialize the retriever with the documents
    retriever = BM25Retriever.from_documents(documents, k=k, **kwargs)
    return retriever.get_relevant_documents(query)

###### LLM Reraner functions ######

def expected_relevance_values(logits, grades_token_ids, list_grades):
    next_token_logits = logits[:, -1, :]
    next_token_logits = next_token_logits.cpu()[0]
    probabilities = F.softmax(next_token_logits[grades_token_ids], dim=-1).numpy()
    return np.dot(np.array(list_grades), probabilities)

"""
def peak_relevance_likelihood(logits, grades_token_ids, list_grades):
    index_max_grade = np.array(list_grades).argmax()
    next_token_logits = logits[:, -1, :]
    probabilities = F.softmax(next_token_logits, dim=-1).cpu().numpy()[0]
    return probabilities[grades_token_ids[index_max_grade]]
"""
################# ONE-SHOT VERSION ##################################

def llm_reranking(tokenizer, model, query, retrieved_documents, assessing_method, k=10):
    docs_content = [doc.page_content for doc in retrieved_documents]

    scores = []
    for document in docs_content:
        score = assessing_method(tokenizer, model, query, document)
        scores.append(score)

    docs_with_scores = list(zip(retrieved_documents, scores, strict=False))
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_documents = [doc for doc, score in docs_with_scores] # docs_with_scores 
    return sorted_documents[:k]

def compute_sequence_log_probs(tokenizer, model, sequence):
    # Tokenize the input sequence
    inputs = tokenizer(sequence, return_tensors='pt')
    input_ids = inputs['input_ids']

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

def RG_S(tokenizer, model, query, document, k=3):

    list_grades = list(range(k))
    grades_token_ids = [tokenizer(str(grade))["input_ids"][1] for grade in list_grades]
 
    RG_S_template = """
    Sur une échelle de 0 à {k}, jugez la pertinence entre la requête et le document.
    Requête : {query}
    Document : {document}
    Réponse : """ 

    messages = [
        {"role": "system", "content": "Tu es un assistant chatbot expert en Statistique Publique."},
        {"role": "user", "content": RG_S_template.format(query=query, document=document, k=k)},
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = tokenizer(input_text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return expected_relevance_values(logits, grades_token_ids, list_grades)

def RG_4L(tokenizer, model, query, document):
    possible_judgements = [" Parfaitement Pertinent", " Très Pertinent", " Assez Pertinent", " Non Pertinent"]
    list_grades = np.array([3, 2, 1, 0])
    RG_4L_template = """
    Evaluez la pertinence du document donné par rapport à la question posée.
    Répondez uniquement parmi : Parfaitement Pertinent, Très Pertinent, Assez Pertinent ou Non Pertinent.
    Requête : {query}
    Document : {document}
    Réponse : {judgement}"""

    messages = [
        {"role": "system", "content": "Tu es un assistant chatbot expert en Statistique Publique."},
        {"role": "user", "content": RG_4L_template},
    ]

    log_probs = []
    for judgement in possible_judgements:
        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False
        ).format(query=query, document=document, judgement=judgement)
        log_probs.append(compute_sequence_log_probs(tokenizer, model, sequence=input_text))

    probs = F.softmax(torch.tensor(log_probs), dim=-1).numpy()
    return np.dot(probs, list_grades)

def RG_3L(tokenizer, model, query, document):
    possible_judgements = [" Très Pertinent", " Assez Pertinent", " Non Pertinent"]
    list_grades = np.array([2, 1, 0])
    RG_3L_template = """
    Evaluez la pertinence du document donné par rapport à la question posée.
    Répondez uniquement parmi : Très Pertinent, Assez Pertinent ou Non Pertinent.
    Requête : {query}
    Document : {document}
    Réponse : {judgement}"""

    messages = [
        {"role": "system", "content": "Tu es un assistant chatbot expert en Statistique Publique."},
        {"role": "user", "content": RG_3L_template},
    ]

    log_probs = []
    for judgement in possible_judgements:
        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False
        ).format(query=query, document=document, judgement=judgement)
        log_probs.append(compute_sequence_log_probs(tokenizer, model, sequence=input_text))


    probs = F.softmax(torch.tensor(log_probs), dim=-1).numpy()
    return np.dot(probs, list_grades) # expected relevance value
    
def RG_YN(tokenizer, model, query, document):
    list_judgements = [" Oui", " Non"]
    grades_token_ids = [tokenizer(j)["input_ids"][1] for j in list_judgements]
    list_grades = [1, 0]

    RG_YN_template = """
    Pour la requête et le document suivants, jugez s'ils sont pertinents. Répondez UNIQUEMENT par Oui ou Non.
    Requête : {query}
    Document : {document}
    Réponse : """

    messages = [
        {"role": "system", "content": "Tu es un assistant chatbot expert en Statistique Publique."},
        {"role": "user", "content": RG_YN_template.format(query=query, document=document)},
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = tokenizer(input_text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return expected_relevance_values(logits, grades_token_ids, list_grades)


################# BATCH VERSION ##################################

def llm_reranking_batch(tokenizer, model, query, retrieved_documents, assessing_method, k):
    docs_content = [doc.page_content for doc in retrieved_documents]
    batch_size = 25
    scores = []
    for i in range(0, len(docs_content), batch_size):
        # Process documents in batches
        batch_docs = docs_content[i:i + batch_size]
        batch_scores = assessing_method(tokenizer, model, query, batch_docs)
        scores.extend(batch_scores)

    docs_with_scores = list(zip(retrieved_documents, scores, strict=False))
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_documents = [doc for doc, score in docs_with_scores]
    return sorted_documents[:k]

def compute_batch_sequence_log_probs(tokenizer, model, sequences):
    # Tokenize the input sequences
   
    batch_inputs = tokenizer(sequences, return_tensors='pt', padding=True).to(model.device)
    input_ids = batch_inputs['input_ids']

    # Move input tensors to the same device as the model
    inputs = batch_inputs.to(model.device)

    # Get the logits from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Calculate the log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Compute the probability of each sequence in the batch
    batch_sequence_probabilities = []
    for batch_idx in range(len(sequences)):
        sequence_probability = 0.0
        for i in range(len(input_ids[batch_idx]) - 1):
            token_id = input_ids[batch_idx][i + 1]
            sequence_probability += log_probs[batch_idx, i, token_id].item()
        batch_sequence_probabilities.append(sequence_probability)
    return batch_sequence_probabilities

def RG_S_batch(tokenizer, model, query, documents, k=3):
    list_grades = list(range(k))
    grades_token_ids = [tokenizer(str(grade))["input_ids"][1] for grade in list_grades]

    RG_S_template = """
    Sur une échelle de 0 à {k}, jugez la pertinence entre la requête et le document.
    Requête : {query}
    Document : {document}
    Réponse : """

    messages = [
        {"role": "system", "content": "Tu es un assistant chatbot expert en Statistique Publique."}
    ]

    # Create batch inputs by combining the query with each document
    batch_inputs = []
    for document in documents:
        message = messages + [{"role": "user", "content": RG_S_template.format(query=query, document=document, k=k)}]
        input_text = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False
        )
        batch_inputs.append(input_text)

    # Tokenize the batch of inputs
    inputs = tokenizer(batch_inputs, return_tensors='pt', padding=True).to(model.device)

    # Perform inference in a single batch
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Calculate the relevance scores for each document in the batch
    scores = []
    for i in range(logits.size(0)):
        logit = logits[i].unsqueeze(0)
        score = expected_relevance_values(logit, grades_token_ids, list_grades)
        scores.append(score)
    return scores

def RG_YN_batch(tokenizer, model, query, documents):
    list_judgements = [" Oui", " Non"]
    grades_token_ids = [tokenizer(j)["input_ids"][1] for j in list_judgements]
    list_grades = [1, 0]

    RG_YN_template = """
    Pour la requête et le document suivants, jugez s'ils sont pertinents. Répondez UNIQUEMENT par Oui ou Non.
    Requête : {query}
    Document : {document}
    Réponse : """

    batch_inputs = []
    for doc in documents:
        messages = [
            {"role": "system", "content": "Tu es un assistant chatbot expert en Statistique Publique."},
            {"role": "user", "content": RG_YN_template.format(query=query, document=doc)},
        ]
        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        batch_inputs.append(input_text)

    inputs = tokenizer(batch_inputs, return_tensors='pt', padding=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Calculate the relevance scores for each document in the batch
    scores = []
    for i in range(logits.size(0)):
        logit = logits[i].unsqueeze(0)
        score = expected_relevance_values(logit, grades_token_ids, list_grades)
        scores.append(score)
    return scores

def RG_3L_batch(tokenizer, model, query, documents):
    possible_judgements = [" Très Pertinent", " Assez Pertinent", " Non Pertinent"]
    list_grades = np.array([2, 1, 0])
    RG_3L_template = """
    Evaluez la pertinence du document donné par rapport à la question posée.
    Répondez uniquement parmi : Très Pertinent, Assez Pertinent ou Non Pertinent.
    Requête : {query}
    Document : {document}
    Réponse : {judgement}"""

    messages = [
        {"role": "system", "content": "Tu es un assistant chatbot expert en Statistique Publique."},
        {"role": "user", "content": RG_3L_template},
    ]
    # Create batch inputs by combining the query with each document
    scores = []
    for document in documents:
        batch_inputs = []
        for judgement in possible_judgements:

            input_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False
            ).format(query=query, document=document, judgement=judgement)

            batch_inputs.append(input_text)

        # Compute log probabilities for the batch of inputs
        judgements_log_probs = compute_batch_sequence_log_probs(tokenizer, model, sequences=batch_inputs)
        # Calculate softmax probabilities
        probs = F.softmax(torch.tensor(judgements_log_probs), dim=-1).numpy()
        # Compute the expected relevance score
        score = np.dot(probs, list_grades)  # expected relevance value
        scores.append(score)
    
    return scores 

def RG_4L_batch(tokenizer, model, query, documents):
    possible_judgements = [" Parfaitement Pertinent", " Très Pertinent", " Assez Pertinent", " Non Pertinent"]
    list_grades = np.array([3, 2, 1, 0])
    RG_4L_template = """
    Evaluez la pertinence du document donné par rapport à la question posée.
    Répondez uniquement parmi : Parfaitement Pertinent, Très Pertinent, Assez Pertinent ou Non Pertinent.
    Requête : {query}
    Document : {document}
    Réponse : {judgement}"""


    messages = [
        {"role": "system", "content": "Tu es un assistant chatbot expert en Statistique Publique."},
        {"role": "user", "content": RG_4L_template},
    ]
    # Create batch inputs by combining the query with each document
    scores = []
    for document in documents:
        batch_inputs = []
        for judgement in possible_judgements:

            input_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False
            ).format(query=query, document=document, judgement=judgement)

            batch_inputs.append(input_text)

        # Compute log probabilities for the batch of inputs
        judgements_log_probs = compute_batch_sequence_log_probs(tokenizer, model, sequences=batch_inputs)
        # Calculate softmax probabilities
        probs = F.softmax(torch.tensor(judgements_log_probs), dim=-1).numpy()
        # Compute the expected relevance score
        score = np.dot(probs, list_grades)  # expected relevance value
        scores.append(score)
    
    return scores 










"""
def compute_proba_judgement(sequence, judgement):
    # Tokenize the input sequence and judgement
    inputs = tokenizer(sequence, return_tensors='pt')
    input_ids = inputs['input_ids']
    judgement_ids = tokenizer(judgement, return_tensors='pt')['input_ids'][0][1:]
    
    # print("Input IDs:", input_ids)
    # print("Judgement IDs:", judgement_ids)
    
    # Move input tensors to the same device as the model
    input_ids = input_ids.to(model.device)
    
    # Get the logits from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Calculate the probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)

    # Convert input IDs tensor to list for indexing
    input_ids_list = input_ids[0].tolist()[1:]
    judgement_ids_list = judgement_ids.tolist()

    start_index  = find_sublist_indices(main_list=input_ids_list, sublist=judgement_ids_list)
    #print("Start index:", start_index, "End index:", end_index)

    if start_index == -1 or end_index == -1:
        raise ValueError("Judgement sublist not found in the input sequence")

    # Compute the probability of the sequence
    sequence_probability = 0.0
    list_indexes  = list(range(0, start_index + len(judgement_ids_list), 1))
    for i in list_indexes:
        token_id = input_ids_list[i] 
        token_log_prob = log_probs[0, i, token_id].item()
        token_prob = probs[0, i, token_id].item()
        print("     Token:", tokenizer.decode([token_id]),"/ Probability:", token_prob * 100, "%")
        sequence_probability += token_log_prob

    return sequence_probability/len(list_indexes)
"""