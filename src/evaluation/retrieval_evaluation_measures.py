import numpy as np 


class RetrievalEvaluationMeasure:
    pass

## Measures #############

def recall(retrieved, relevant):
    intersection = set(retrieved) & set(relevant)
    return np.round(len(intersection) / len(relevant), 3)  if len(relevant) > 0 else 0

def precision(retrieved, relevant):
    intersection = set(retrieved) & set(relevant)
    return np.round(len(intersection) / len(retrieved), 3)  if len(retrieved) > 0 else 0

def hit_rate(retrieved, relevant):
    """ 
    Hit rate metric is equivalent to the accuracy 
    """
    correct_retrieved = sum(1 for pred, label in zip(retrieved, relevant) if pred == label)
    total_retrieved = len(retrieved)
    hit_rate = correct_retrieved / total_retrieved
    return hit_rate

def mmr(retrieved, relevant):
    # compute Mean Reciprocal Rank (Order Aware Metrics)
    if relevant not in retrieved:
        mrr_score = 1/np.inf
    else: 
        rank_q = retrieved.index(relevant)
        mrr_score = 1/(rank_q + 1)
    return mrr_score

