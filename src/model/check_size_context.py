import numpy as np
from langchain_core.prompts import PromptTemplate
from loguru import logger

from src.vectordatabase.output_parsing import format_docs

from .tokenizer import tokenize_prompt


def check_size_context(
    system_instructions: str,
    question_instructions: str,
    question: str,
    best_documents: list[str],
    api_url: str,
    model_name: str,
) -> tuple[str, str]:
    """
    Ensures the combined prompt (system + question + context) fits within the model's token limit.

    If the full context exceeds the model's max token length, the list of documents is truncated
    to fit within the allowed size. If even a single document exceeds the max size, raises ValueError.

    Args:
        system_instructions (str): System-level prompt instructions.
        question_instructions (str): Instructions guiding the question format.
        question (str): The user's question.
        best_documents (list[str]): List of candidate documents to include as context.
        api_url (str): Root URL of the tokenization API.
        model_name (str): Model identifier used by the tokenizer API.

    Returns:
        tuple[str, str]: A tuple of (final context string, full prompt with question and context).

    Raises:
        ValueError: If no document can fit under the model's max token length.
    """
    # Format the documents into a context string
    context = format_docs(best_documents)

    # Create a prompt template and fill it with question and context
    prompt = PromptTemplate.from_template(question_instructions)
    question_with_context = prompt.format(question=question, context=context)

    # Tokenize the entire composed prompt
    result = tokenize_prompt(question_with_context, model_name, api_url)

    if result["count"] > result["max_model_len"]:
        # Tokenize each document individually to determine cumulative length
        length_documents = [
            tokenize_prompt(prompt.format(question=question, context=doc), model_name, api_url)["count"]
            for doc in best_documents
        ]

        # Identify how many documents can fit within the model's limit
        cumsum = np.cumsum(length_documents)
        max_size_context = np.where(cumsum <= result["max_model_len"])[0]

        if len(max_size_context) == 0:
            message = (
                f"Context too long (even one doc exceeds max size).\n"
                f"First document size: {length_documents[0]} tokens vs "
                f"{result['max_model_len']} allowed"
            )
            logger.error(message)
            raise ValueError(message)
        else:
            max_index = max_size_context[-1]
            message = f"Context too long: reducing context to {max_index + 1} document(s)"
            logger.info(message)

            # Rebuild context and prompt with truncated documents
            best_documents = best_documents[: max_index + 1]
            context = format_docs(best_documents)
            question_with_context = prompt.format(question=question, context=context)
    else:
        message = (
            f"Documents added as context ({result['count']} tokens) "
            f"fit into context length ({result['max_model_len']} tokens)"
        )
        logger.success(message)

    return context, question_with_context
