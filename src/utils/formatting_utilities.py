from langchain_core.prompts import PromptTemplate

from src.config import Configurable, DefaultFullConfig, FullConfig


def str_to_bool(value):
    value = value.lower()
    if value == "true":
        return True
    elif value == "false":
        return False
    else:
        raise ValueError(f"Invalid value: {value}")


@Configurable()
def get_chatbot_template(
    system_instruction: str | None = None, config: FullConfig = DefaultFullConfig()
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": system_instruction or config.CHATBOT_SYSTEM_INSTRUCTION,
        },
    ]


def add_sources_to_messages(message: str, sources: list, titles: list, topk: int = 5):
    """
    Append a list of sources and titles to a Chainlit message.

    Args:
    - message (str): The Chainlit message content to which the sources and titles will be added.
    - sources (list): A list of sources to append to the message.
    - titles (list): A list of titles to append to the message.
    - topk (int) : number of displayed sources.
    """
    if len(sources) == len(titles):
        sources_titles = [
            f"{i + 1}. {title} ({source})"
            for i, (source, title) in enumerate(zip(sources, titles, strict=False))
            if i < topk
        ]
        formatted_sources = f"\n\nSources (Top {topk}):\n" + "\n".join(sources_titles)
        message += formatted_sources
    else:
        message += "\n\nNo Sources available"

    return message


def format_docs(docs: list):
    return "\n\n".join(
        [
            f"""
            Doc {i + 1}:\nTitle: {doc.metadata.get("Header 1")}\n
            Source: {doc.metadata.get("url")}\n
            Content:\n{doc.page_content}
            """
            for i, doc in enumerate(docs)
        ]
    )


def create_prompt_from_instructions(system_instructions: str, question_instructions: str) -> PromptTemplate:
    template = f"""
    {system_instructions}

    {question_instructions}
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    return custom_rag_prompt
