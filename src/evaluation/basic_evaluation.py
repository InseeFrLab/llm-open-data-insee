import openai
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores.base import VectorStoreRetriever

from src.vectordatabase.output_parsing import format_docs, langchain_documents_to_df

with open("./prompt/question.md", encoding="utf-8") as f:
    default_question_prompt = f.read()

with open("./prompt/system.md", encoding="utf-8") as f:
    default_system_prompt = f.read()

with open("./prompt/system_no_context.md", encoding="utf-8") as f:
    default_system_prompt_no_context = f.read()


default_model = "mistralai/Mistral-Small-24B-Instruct-2501"


def compare_retrieved_from_ground_truth(
    retriever: VectorStoreRetriever, question: str, valid_urls: list[str]
) -> pd.DataFrame:
    """
    Compare documents retrieved by a retriever against a ground truth set of valid URLs.

    This function queries a retriever with a specific question, converts the retrieved documents
    into a DataFrame, and annotates each document with whether its URL is part of the expected ground truth.
    Additional metadata such as the total number of expected pages and the original question is also added.

    Args:
        retriever (VectorStoreRetriever): The retriever instance used to fetch relevant documents.
        question (str): The question to query the retriever.
        valid_urls (List[str]): A list of URLs considered as ground truth for validation.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved documents, including validation flags
                      and additional metadata for comparison.
    """
    retrieved_docs = retriever.invoke(question)

    if isinstance(retrieved_docs, dict):
        retrieved_docs = retrieved_docs["context"]

    # Transform in pd.DataFrame to ease question by question comparison
    result_retriever_raw = langchain_documents_to_df(retrieved_docs)

    result_retriever_raw["url_expected"] = result_retriever_raw["url"].isin(valid_urls)

    result_retriever_raw["number_pages_expected"] = len(valid_urls)
    result_retriever_raw["question"] = question

    return result_retriever_raw


def collect_answers_retrievers(
    retriever: VectorStoreRetriever,
    question: str,
    valid_urls: str | list[str],
    with_generation: bool = False,
    chat_client: openai.OpenAI = None,
    chat_client_options: dict = None,
) -> tuple[pd.DataFrame, VectorStoreRetriever, bool | str]:
    """
    Collect retrieved documents for a given question and optionally generate an answer using a chat model.

    This function queries a retriever with a question, compares the retrieved documents to a ground truth
    list of valid URLs, and optionally uses an OpenAI chat client to generate an answer based on the retrieved context.
    Chat client options such as model, system prompt, and question prompt can be customized via `chat_client_options`.

    Args:
        retriever (VectorStoreRetriever): The retriever instance used to fetch relevant documents.
        question (str): The question to query the retriever.
        valid_urls (Union[str, List[str]]): A single URL or a list of URLs representing the ground truth for evaluation.
        with_generation (bool, optional): If True, generates an answer using the chat client. Defaults to False.
        chat_client (openai.OpenAI, optional): An instance of the OpenAI client used for answer generation. Required if `with_generation` is True.
        chat_client_options (dict, optional): Dictionary containing chat model options such as:
            - "model" (str): The model name to use (e.g., "gpt-4").
            - "question_prompt" (str): Template to format the question and context.
            - "system_prompt" (str): System-level prompt guiding the model's behavior.
            Defaults to a predefined configuration if not provided.

    Returns:
        Tuple:
            - pd.DataFrame: DataFrame comparing retrieved documents against ground truth.
            - VectorStoreRetriever: The raw retrieved documents.
            - Union[bool, str]: False if no generation is requested, otherwise the generated response text.
    """

    if isinstance(valid_urls, str):
        valid_urls = [valid_urls]

    if with_generation is True and chat_client is None:
        raise ValueError("chat_client must be provided when with_generation is True")

    default_options = {
        "model": default_model,
        "question_prompt": default_question_prompt,
        "system_prompt": default_system_prompt,
        "system_prompt_no_context": default_system_prompt_no_context
    }

    if chat_client_options is None:
        chat_client_options = default_options
    else:
        # Fill missing keys from default
        for key, value in default_options.items():
            chat_client_options.setdefault(key, value)

    retrieved_docs = retriever.invoke(question)
    docs_vs_truth = compare_retrieved_from_ground_truth(retriever, question, valid_urls)

    if with_generation is False:
        return docs_vs_truth, retrieved_docs, False

    prompt = PromptTemplate.from_template(chat_client_options.get("question_prompt"))
    context = format_docs(retrieved_docs)
    question_with_context = prompt.format(question=question, context=context)

    response = chat_client.chat.completions.create(
        model=chat_client_options.get("model"),
        messages=[
            {"role": "system", "content": chat_client_options.get("system_prompt")},
            {"role": "user", "content": question_with_context},
        ],
        stream=False,
    )
    response_no_context = chat_client.chat.completions.create(
        model=chat_client_options.get("model"),
        messages=[
            {"role": "system", "content": chat_client_options.get("system_prompt_no_context")},
            {"role": "user", "content": question},
        ],
        stream=False,
    )

    response = response.choices[0].message.content
    response_no_context = response_no_context.choices[0].message.content

    return docs_vs_truth, retrieved_docs, response, response_no_context


def compare_retriever_with_expected_docs(
    retriever: VectorStoreRetriever,
    ground_truth_df: pd.DataFrame,
    question_col: str,
    ground_truth_col: str,
    with_generation: bool = False,
    chat_client: openai.OpenAI = None,
    chat_client_options: dict = None,
) -> tuple[pd.DataFrame, list[str | bool]]:
    """
    Compare retriever outputs with expected documents from a ground truth DataFrame.

    For each question in the ground truth DataFrame, this function retrieves documents
    using the retriever, evaluates them against the expected documents, and optionally
    generates an answer using a language model, with configurable chat client options.

    Args:
        retriever (VectorStoreRetriever): The retriever used to fetch relevant documents.
        ground_truth_df (pd.DataFrame): DataFrame containing the ground truth questions and expected documents.
        question_col (str): Column name in the DataFrame containing the questions.
        ground_truth_col (str): Column name in the DataFrame containing the expected documents or URLs.
        with_generation (bool, optional): Whether to generate an answer using a language model. Defaults to False.
        chat_client (openai.OpenAI, optional): An instance of OpenAI client used for answer generation. Required if `with_generation` is True.
        chat_client_options (dict, optional): Dictionary with options for chat client, such as:
            - "model" (str): The model name (e.g., "gpt-4").
            - "question_prompt" (str): Template for formatting question + context.
            - "system_prompt" (str): System-level prompt for guiding the model behavior.
            If None, default values will be used.

    Returns:
        Tuple:
            - pd.DataFrame: Concatenated DataFrame of retrieved document comparisons.
            - List[Union[str, bool]]: List of generated responses or False if no generation was requested.
    """
    answers_retriever = []
    answers_generative = []
    answers_generative_no_context = []

    for _idx, faq_items in ground_truth_df.iterrows():
        docs, _, response, response_no_context  = collect_answers_retrievers(
            retriever=retriever,
            question=faq_items[question_col],
            valid_urls=faq_items[ground_truth_col],
            with_generation=with_generation,
            chat_client=chat_client,
            chat_client_options=chat_client_options,
        )
        answers_retriever.append(docs)
        answers_generative.append(response)
        answers_generative_no_context.append(response_no_context)

    answers_retriever = pd.concat(answers_retriever)

    return answers_retriever, answers_generative, answers_generative_no_context


# OLD ----------------------------


def transform_answers_bot(answers_bot: pd.DataFrame, k: int):
    # Accurate page or document response
    nbre_documents_answers_bot = (
        answers_bot.groupby("question")
        .agg({"url_expected": "sum", "number_pages_expected": "mean"})
        .sort_values("url_expected", ascending=False)
    )

    nbre_pages_answers_bot = (
        answers_bot.drop_duplicates(subset=["question", "url"])
        .groupby("question")
        .agg({"url_expected": "sum", "number_pages_expected": "mean"})
        .sort_values("url_expected", ascending=False)
    )

    eval_reponses_bot = (
        nbre_pages_answers_bot.merge(nbre_documents_answers_bot, on=["question", "number_pages_expected"])
        .rename(
            columns={
                "url_expected_x": "Nombre de pages citées par le bot qui sont OK",
                "url_expected_y": "Nombre de documents cités par le bot qui sont dans la bonne page",
                "number_pages_expected": "Nombre de pages citées dans la réponse de la FAQ",
            }
        )
        .reset_index()
    )

    # Top k answers
    answers_bot["cumsum_url_expected"] = answers_bot.groupby(["question"])["url_expected"].cumsum()
    answers_bot["document_position"] = answers_bot.groupby("question").cumcount() + 1
    answers_bot["cumsum_url_expected"] = answers_bot["cumsum_url_expected"].clip(upper=1)
    answers_bot["cumsum_url_expected"] = answers_bot["cumsum_url_expected"].astype(bool)

    answers_bot_topk = answers_bot.groupby("document_position").agg({"cumsum_url_expected": "mean"}).reset_index()
    answers_bot_topk = answers_bot_topk.loc[answers_bot_topk["document_position"] <= k]

    return eval_reponses_bot, answers_bot_topk


def _answer_faq_by_bot(retriever, question, valid_urls):
    retrieved_docs = retriever.invoke(question)
    if isinstance(retrieved_docs, dict):
        retrieved_docs = retrieved_docs["context"]
    result_retriever_raw = langchain_documents_to_df(retrieved_docs)
    result_retriever_raw["url_expected"] = result_retriever_raw["url"].isin(valid_urls)
    result_retriever_raw["number_pages_expected"] = len(valid_urls)
    result_retriever_raw["question"] = question
    return result_retriever_raw


def answer_faq_by_bot(retriever, faq):
    answers_bot = []
    for _idx, faq_items in faq.iterrows():
        answers_bot.append(_answer_faq_by_bot(retriever, faq_items["titre"], faq_items["urls"].split(", ")))
    answers_bot = pd.concat(answers_bot)
    return answers_bot
