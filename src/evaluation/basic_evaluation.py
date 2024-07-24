import pandas as pd
from src.db_building import chroma_topk_to_df


def transform_answers_bot(answers_bot: pd.DataFrame, k: int):
    # Accurate page or document response
    nbre_documents_answers_bot = (answers_bot
        .groupby("question")
        .agg({'url_expected': 'sum', 'number_pages_expected': 'mean'})
        .sort_values("url_expected", ascending=False)
    )
    
    nbre_pages_answers_bot = (
        answers_bot.drop_duplicates(subset=['question', 'url'])
        .groupby("question")
        .agg({'url_expected': 'sum', 'number_pages_expected': 'mean'})
        .sort_values("url_expected", ascending=False)
    )
    
    eval_reponses_bot = (
        nbre_pages_answers_bot.merge(
            nbre_documents_answers_bot,
            on=["question", "number_pages_expected"]
        )
        .rename(
            columns={
                "url_expected_x": "Nombre de pages citées par le bot qui sont OK",
                "url_expected_y": "Nombre de documents cités par le bot qui sont dans la bonne page",
                "number_pages_expected": "Nombre de pages citées dans la réponse de la FAQ"
            }
        )
        .reset_index()
    )
    
    # Top k answers
    answers_bot["cumsum_url_expected"] = (
        answers_bot
        .groupby(['question'])['url_expected']
        .cumsum()
    )
    answers_bot["document_position"] = answers_bot.groupby('question').cumcount() + 1
    answers_bot["cumsum_url_expected"] = answers_bot["cumsum_url_expected"].clip(upper=1)
    answers_bot["cumsum_url_expected"] = answers_bot["cumsum_url_expected"].astype(bool)
    
    answers_bot_topk = (
        answers_bot
        .groupby("document_position")
        .agg({"cumsum_url_expected": "mean"})
        .reset_index()
    )
    answers_bot_topk = answers_bot_topk.loc[answers_bot_topk['document_position'] <= k]
    
    return eval_reponses_bot, answers_bot_topk


def _answer_faq_by_bot(retriever, question, valid_urls):
    
    retrieved_docs = retriever.invoke(
        question
    )
    result_retriever_raw = chroma_topk_to_df(retrieved_docs)
    result_retriever_raw['url_expected'] = result_retriever_raw['url'].isin(valid_urls)
    result_retriever_raw['number_pages_expected'] = len(valid_urls)
    result_retriever_raw['question'] = question
    return result_retriever_raw


def answer_faq_by_bot(retriever, faq):
    answers_bot = []
    for idx, faq_items in faq.iterrows():
        answers_bot.append(
            answer_faq_by_bot(
                retriever, faq_items['titre'], faq_items['urls'].split(", ")
            )
        )
    answers_bot = pd.concat(answers_bot)
    return answers_bot