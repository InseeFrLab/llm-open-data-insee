from langchain_core.prompts import PromptTemplate


def create_prompt_from_instructions(system_instructions: str, question_instructions: str) -> PromptTemplate:
    template = f"""
    {system_instructions}

    {question_instructions}
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    return custom_rag_prompt


def format_docs(docs: list):
    docs_in_prompt = "\n\n".join(
        [
            f"""--------------\n
Doc {i + 1}:\n
Title: {doc.metadata.get("titre", "").replace("#", "")}\n
Abstract: {doc.metadata.get("abstract", "")}\n
Source: {doc.metadata.get("url", "")}\n
Content:\n{doc.page_content or "[No content]"}\n
"""
            for i, doc in enumerate(docs)
        ]
    )
    return docs_in_prompt

