import logging

import pandas as pd
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from transformers import AutoTokenizer

from .utils_db import parse_xmls

logger = logging.getLogger(__name__)

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]


def chunk_documents(
    data: pd.DataFrame,
    embedding_model: str,
    markdown_split: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[Document]:
    """
    Chunks documents from a dataframe into smaller pieces using specified tokenizer settings
    or custom settings.

    Parameters:
    - data (pd.DataFrame): The dataframe containing documents to be chunked.
    - embedding_model (str): The name of the Hugging Face tokenizer to use.
    - markdown_split (bool): Whether to split markdown headers into separate chunks.
    - chunk_size (int, optional): Size of each chunk if not using hf_tokenizer.
    - chunk_overlap (int, optional): Overlap size between chunks if not using hf_tokenizer.
    - separators (list, optional): List of separators to use for splitting the text.

    Returns:
    - list[Document]: The list of processed unique document chunks
    """

    logging.info("Building the list of document objects")

    # advantage of using a loader
    # No need to know which metadata are stored in the dataframe
    # Every column except page_content_column contains metadata
    document_list = DataFrameLoader(data, page_content_column="content").load()

    if markdown_split:
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)
        document_list = make_md_splits(document_list, markdown_splitter)

    # Load the tokenizer
    autokenizer = AutoTokenizer.from_pretrained(embedding_model)

    if autokenizer.model_max_length is not None:
        chunk_size = autokenizer.model_max_length

    # Initialize token splitter
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        autokenizer,
        chunk_size=chunk_size,
        # chunk_overlap=chunk_overlap,
        separators=separators,
    )

    # Split documents into chunks
    docs_processed = text_splitter.split_documents(document_list)

    # Remove duplicates
    unique_texts = set()
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts.add(doc.page_content)
            docs_processed_unique.append(doc)

    logging.info(f"Number of created chunks: {len(docs_processed_unique)} in the Vector Database")

    return docs_processed_unique


def make_md_splits(document_list: list[Document], markdown_splitter: MarkdownHeaderTextSplitter) -> list[Document]:
    """
    Splits the content of each document in the document list based on Markdown headers,
    and preserves the original metadata in each split section.

    Args:
        document_list (list[Document]): List of Document objects to be split.
        markdown_splitter (MarkdownHeaderTextSplitter): An instance of MarkdownHeaderTextSplitter
            used to perform the text splitting based on Markdown headers.

    Returns:
        list[Document]: List of split Document objects with updated metadata.
    """
    splitted_docs = []

    for doc in document_list:
        # Split the document content into sections based on Markdown headers
        md_header_splits = markdown_splitter.split_text(doc.page_content)

        for md_section in md_header_splits:
            # Update metadata for each split section with the original document's metadata
            md_section.metadata.update(doc.metadata)
            splitted_docs.append(md_section)

    return splitted_docs


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
        ],
    ]

    df = df.fillna(value="")

    return df


def parse_transform_documents(
    data: pd.DataFrame,
    max_document_size: int | None = None,
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

    # Optionally split documents into smaller chunks
    if max_document_size:
        logger.debug(f"Splitting documents if they exceed {max_document_size} tokens")
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=max_document_size, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)

    return documents
