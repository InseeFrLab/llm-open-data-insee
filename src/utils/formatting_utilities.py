from collections.abc import Mapping
from typing import Any

from src.config import default_config


def str_to_bool(value):
    value = value.lower()
    if value == "true":
        return True
    elif value == "false":
        return False
    else:
        raise ValueError(f"Invalid value: {value}")


def get_chatbot_template(
    system_instruction: str | None = None, config: Mapping[str, Any] = default_config
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": system_instruction or config["CHATBOT_SYSTEM_INSTRUCTION"],
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
            f"{i+1}. {title} ({source})"
            for i, (source, title) in enumerate(zip(sources, titles, strict=False))
            if i < topk
        ]
        formatted_sources = f"\n\nSources (Top {topk}):\n" + "\n".join(sources_titles)
        message += formatted_sources
    else:
        message += "\n\nNo Sources available"

    return message
