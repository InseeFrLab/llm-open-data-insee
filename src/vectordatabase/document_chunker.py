import logging

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents.base import Document

from .utils_db import parse_xmls

logger = logging.getLogger(__name__)


def chunk_documents(documents: list[Document], **kwargs) -> list[Document]:
    logging.info("Building the list of document objects")
    logging.info(f"The following parameters have been applied: {kwargs}")

    # Initialize token splitter
    docs_processed = RecursiveCharacterTextSplitter(**kwargs).split_documents(documents)

    logging.info(f"Number of created chunks: {len(docs_processed)} in the Vector Database")

    return docs_processed


def _parser_xml_web4g(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Parsing XML content")
    parsed_pages = parse_xmls(data)

    # Merge parsed XML data with the original DataFrame
    df = (
        data.set_index("id")
        .merge(pd.DataFrame(parsed_pages), left_index=True, right_index=True)
        .drop(columns=["xml_content"], errors="ignore")  # Drop only if exists
    )

    df = df.loc[
        :,
        [
            "titre",
            "categorie",
            "url",
            "dateDiffusion",
            "theme",
            "collection",
            "libelleAffichageGeo",
            "content",
            "abstract",
        ],
    ]

    df = df.fillna(value="")

    return df


def parse_documents(
    data: pd.DataFrame,
    page_column_content: str = "content",
    engine_output: str = "pandas",
) -> pd.DataFrame | list:
    """
    Parses and transforms document data, optionally splitting documents into smaller chunks.

    This function processes a DataFrame containing document data by:
    1. Parsing XML content.
    2. Merging parsed content into the original DataFrame.
    3. Converting the data into LangChain document format if needed.
    4. Optionally splitting documents into smaller chunks if `max_document_size` is specified.

    Args:
        data (pd.DataFrame): Input DataFrame containing documents, with an 'id' column.
        max_document_size (Optional[int], optional): The maximum number of tokens per document chunk.
            If `None`, documents are not split. Defaults to `None`.
        page_column_content (str, optional): The column containing document content. Defaults to "content".
        engine_output (str, optional): The output format, either `"pandas"` or `"langchain"`.
            Defaults to `"pandas"`.

    Returns:
        Union[pd.DataFrame, list]: A DataFrame if `engine_output="pandas"`, or a list of LangChain documents.

    Raises:
        ValueError: If `engine_output` is not `"pandas"` or `"langchain"`.
    """

    if engine_output not in {"pandas", "langchain"}:
        raise ValueError("'engine_output' should be 'pandas' or 'langchain'")

    df = _parser_xml_web4g(data)

    if engine_output == "pandas":
        logger.debug("Returning Pandas DataFrame since 'engine_output' is set to 'pandas'")
        return df

    logger.debug("Transforming into LangChain document format")

    # Load DataFrame into LangChain document format
    loader = DataFrameLoader(df, page_content_column=page_column_content)
    documents = loader.load()

    return documents
