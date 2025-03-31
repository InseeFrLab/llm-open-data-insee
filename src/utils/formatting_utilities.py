from langchain_core.prompts import PromptTemplate


def create_prompt_from_instructions(system_instructions: str, question_instructions: str) -> PromptTemplate:
    template = f"""
    {system_instructions}

    {question_instructions}
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    return custom_rag_prompt


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
