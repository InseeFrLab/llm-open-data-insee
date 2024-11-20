def compare_performance_reranking(eval_reponses_bot_after_reranker, eval_reponses_bot_before_reranker):
    comparaison_eval_reponses = eval_reponses_bot_after_reranker.merge(
        eval_reponses_bot_before_reranker,
        on=["question", "Nombre de pages citées dans la réponse de la FAQ"],
        suffixes=[" (after reranking) ", " (before reranking) "],
    )

    # Reordering columns
    columns = comparaison_eval_reponses.columns.sort_values()
    question_cols = ["question"]
    pages_cited_cols = [col for col in columns if col.startswith("Nombre de pages citées")]
    documents_cited_cols = [col for col in columns if col.startswith("Nombre de documents cités")]

    # Combine all lists in the desired order
    new_order = question_cols + pages_cited_cols + documents_cited_cols
    comparaison_eval_reponses = comparaison_eval_reponses.loc[:, new_order]
    return comparaison_eval_reponses
