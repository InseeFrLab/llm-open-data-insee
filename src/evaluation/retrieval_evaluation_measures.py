# retrieval_evaluation_measures.py
import numpy as np


class RetrievalEvaluationMeasure:
    ## Measures #############

    def recall(self, retrieved, relevant):
        intersection = set(retrieved) & set(relevant)
        return (
            np.round(len(intersection) / len(relevant), 3) if len(relevant) > 0 else 0
        )

    def precision(self, retrieved, relevant):
        intersection = set(retrieved) & set(relevant)
        return (
            np.round(len(intersection) / len(retrieved), 3) if len(retrieved) > 0 else 0
        )

    def hit_rate(self, retrieved, relevant):
        """
        Hit rate metric is equivalent to the accuracy
        """
        correct_retrieved = sum(1 for doc in retrieved if doc in relevant)
        total_retrieved = len(retrieved)
        hit_rate = correct_retrieved / total_retrieved
        return hit_rate

    def mrr(self, retrieved, relevant):
        """
        compute Mean Reciprocal Rank (Order Aware Metrics)
        """
        mrr_score = 0.0
        for rank, doc in enumerate(retrieved):
            if doc in relevant:
                mrr_score = 1 / (rank + 1)
                break
        return mrr_score

    def relevance_score(self, retrieved, relevant):
        return [
            1 if retrieved_source in relevant else 0 for retrieved_source in retrieved
        ]

    def dcg(self, relevance_scores, k=None):
        if k is None:
            k = len(relevance_scores)
        relevance_scores = np.asfarray(relevance_scores)[:k]
        return np.sum(
            relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2))
        )

    def idcg(self, relevance_scores, k=None):
        if k is None:
            k = len(relevance_scores)
        sorted_scores = sorted(relevance_scores, reverse=True)
        return self.dcg(sorted_scores, k)

    def ndcg(self, retrieved, relevant, k=None):
        relevance_scores = self.relevance_score(retrieved, relevant)
        actual_dcg = self.dcg(relevance_scores, k)
        ideal_dcg = self.idcg(relevance_scores, k)

        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
