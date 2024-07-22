import logging
import re
from collections.abc import Generator
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag
from markdownify import MarkdownConverter
from tqdm import tqdm

TAGS_TO_IGNORE = [
    "sage",
    "numero",
    "donnees-complementaires",
    "document-imprimable",
    "graphique",
    "fichiers-donnees",
    "dictionnaire-variables",
    "image",
]


def get_soup(xml_string: str) -> BeautifulSoup:
    """
    Parses an XML string and returns a BeautifulSoup object.

    Args:
        xml_string (str): The XML string to be parsed.

    Returns:
        BeautifulSoup: A BeautifulSoup object initialized with the provided XML string.
    """
    soup = BeautifulSoup(xml_string, features="xml")
    return soup


def url_builder(row: pd.Series) -> str:
    """
    Constructs a URL based on the category and id of a given pandas Series row.

    Parameters:
    row (pd.Series): A pandas Series containing 'categorie' and 'id' fields.

    Returns:
    str: Constructed URL if valid category and id are present, otherwise None.
    """
    category = row.get("categorie")
    if category is None:
        return None

    dict_url = {
        "Publications grand public": "statistiques",
        "Communiqués de presse": "information",
        "Chiffres-clés": "statistiques",
        "Chiffres détaillés": "statistiques",
        "Actualités": "statistiques",
        "L'Insee et la statistique publique": "information",
        "Services": "information",
        "Méthodes": "information",
        "Dossiers de presse": "information",
        "Courrier des statistiques": "information",
        "Géographie": "information",
        "Séries chronologiques": "statistiques",
        "Sources": "information",
        "Publications pour expert": "statistiques",
        "Cartes interactives": "",
        "Outils interactifs": "statistiques",
    }

    section = dict_url.get(category)
    if section is None:
        return None

    base_url = "https://www.insee.fr/fr"
    return f"""{base_url}/{section}/{row["id"]}"""


def url_builder_metadata(row: pd.Series) -> str:
    """
    Rebuilds a valid URL for metadata based on the id of a given pandas Series row.

    The base URL looks like: https://www.insee.fr/fr/metadonnees/

    Parameters:
    row (pd.Series): A pandas Series containing an 'id' field.

    Returns:
    str: Constructed metadata URL if the id matches a known pattern, otherwise None.
    """
    base_url = "https://www.insee.fr/fr/metadonnees"
    pattern_source = r"^s"
    pattern_indicator = r"^p"

    row_id = row.get("id")
    if row_id is None:
        return None

    if re.match(pattern_source, row_id):
        return f"{base_url}/source/serie/{row_id}"

    if re.match(pattern_indicator, row_id):
        return f"{base_url}/source/indicateur/{row_id}/description"

    return None


def complete_url_builder(table: pd.DataFrame) -> pd.Series:
    """
    Builds a complete list of URLs for each row in the given DataFrame.

    If the URL cannot be constructed using `url_builder`, it tries `url_builder_metadata`.

    Parameters:
    table (pd.DataFrame): A pandas DataFrame containing rows with necessary fields.

    Returns:
    pd.Series: A pandas Series containing the constructed URLs or None for each row.
    """
    urls = []
    for _, row in tqdm(table.iterrows(), total=table.shape[0], desc="Building URLs"):
        url = url_builder(row)
        if url is None:
            url = url_builder_metadata(row)
        urls.append(url)

    return pd.Series(urls)


def prepend_text_to_tag(tag, text):
    """
    Prepends the given text to the content of the specified tag.

    Parameters:
    tag (Tag): The BeautifulSoup tag object to which the text will be prepended.
    text (str): The text to prepend to the tag's content.

    """
    if tag.string is not None:
        tag.insert(0, text)
    else:
        tag.string = text


def parse_xmls(
    data: pd.DataFrame, id: str = "id", xml_column: str = "xml_content"
) -> pd.DataFrame:
    """
    Parses XML content from a DataFrame, extracts data, formats it, and returns a new DataFrame with the formatted content.

    Args:
        data (pd.DataFrame): The input DataFrame containing XML data.
        id (str, optional): The column name for the unique identifier of each row. Defaults to "id".
        xml_column (str, optional): The column name containing the XML content. Defaults to "xml_content".

    Returns:
        pd.DataFrame: A DataFrame with 'id' as the index and formatted content in the 'content' column.
    """
    parsed_pages = {"id": [], "content": []}

    for i, row in data.iterrows():
        page_id = row[id]
        logging.info(f"Processing page {page_id} -- {i}/{len(data)}")

        if not row[xml_column]:
            # When xml_content is empty, we skip the page
            # TODO: (to be changed, we should extract the xml in the first place)
            continue

        # Extract the xml content and making it html compliant
        soup = format_tags(get_soup(row[xml_column]), TAGS_TO_IGNORE)

        # Transform the xml content into markdown
        parsed_page = md(
            soup,
            escape_misc=False,
            escape_asterisks=False,
            bullets="-",
            heading_style="ATX",
        )

        parsed_pages["id"].append(page_id)
        parsed_pages["content"].append(remove_excessive_newlines(parsed_page))

    return pd.DataFrame(parsed_pages).set_index("id")


def split_list(input_list: list[Any], chunk_size: int) -> Generator[list[Any]]:
    """
    Splits a list into smaller chunks of a specified size.

    Parameters:
    ----------
    input_list : list[Any]
        The list to be split into chunks.
    chunk_size : int
        The size of each chunk.

    Yields:
    -------
    Generator[list[Any]]
        A generator that yields chunks of the input list.

    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    for i in range(0, len(input_list), chunk_size):
        yield input_list[i : i + chunk_size]


def format_tags(soup: Tag, tags_to_ignore: list[str]) -> Tag:
    """
    Formats the tags in the provided BeautifulSoup object according to specific HTML rules.

    Parameters:
    soup (Tag): The BeautifulSoup tag object to format.
    tags_to_ignore (list[str]): A list of tag names to ignore and remove.

    Returns:
    Tag: The formatted BeautifulSoup tag object.
    """
    soup_copy = soup
    TAGS_FIGURE_CHILDREN = ["graphique", "tableau"]

    # Determine if figure tags should be removed
    remove_figure = [tag in tags_to_ignore for tag in TAGS_FIGURE_CHILDREN]

    for tag in soup_copy.find_all():
        # Remove tags to ignore
        if tag.name in tags_to_ignore:
            tag.decompose()
            continue

        # Remove figure tags when they contain tags to ignore
        # We need to do that because some content (e.g titles, sources...) are in the figure tag and not
        # in the children tags (graphique, tableau...)
        # Maybe we still want to keep it ?
        if any(remove_figure) and tag.name == "figure":
            if all(
                tag.find(child_tag) is None
                for child_tag, remove in zip(
                    TAGS_FIGURE_CHILDREN, remove_figure, strict=False
                )
                if not remove
            ):
                tag.decompose()
            continue

        ## HEADINGS
        # Rename titre tags
        if tag.name == "titre" and len(list(tag.parents)) == 2:
            tag.name = "h1"
            continue

        # Rename sous-titre tags
        if tag.name == "sous-titre" and tag.contents:
            tag.name = "h2"
            continue

        # Rename chapo tags
        if tag.name == "chapo":
            prepend_text_to_tag(tag, "Résumé : ")
            tag.name = "h2"
            continue

        # Rename sources tags
        if tag.name == "sources" and len(list(tag.parents)) == 2:
            prepend_text_to_tag(tag, "Sources : ")
            tag.name = "h2"
            continue

        # Rename definition tags
        if tag.name == "definitions" and len(list(tag.parents)) == 2:
            prepend_text_to_tag(tag, "Définitions : ")
            tag.name = "h2"
            continue

        # Rename liens-transverses tags
        if tag.name == "liens-transverses" and len(list(tag.parents)) == 2:
            txt = tag.get("titre") if tag.get("titre") is not None else "Références"
            prepend_text_to_tag(tag, txt)
            tag.name = "h2"
            continue

        # Rename avertissement tags
        if tag.name == "avertissement" and len(list(tag.parents)) == 2:
            tag.name = "h2"
            continue

        # Rename intertitre tags
        if tag.name == "intertitre":
            tag.name = (
                f"h{int(tag.get("niveau")) + 2}"
                if tag.get("niveau") is not None
                else "h3"
            )
            continue

        ## TABLES
        # Rename lignes tags
        if tag.name == "lignes":
            if tag.get("type") == "entete":
                tag.name = "thead"
            elif tag.get("type") == "donnees":
                tag.name = "tbody"
            continue

        # Rename ligne tags
        if tag.name == "ligne":
            tag.name = "tr"
            continue

        # Rename cellule entete tags
        if tag.name == "cellule":
            tag.name = "th" if tag.get("entete") == "colonne" else "td"
            continue

        # Rename tableau tags
        if tag.name == "tableau":
            tag.name = "table"
            continue

        ## TEXT
        # Rename paragraphe tags
        if tag.name == "paragraphe":
            tag.name = "p"
            continue

        # Rename lien-externe tags
        if tag.name == "lien-externe":
            tag.name = "a"
            tag["href"] = tag.get("url")
            if tag.get("url") is not None:
                del tag["url"]
            continue

        # Rename emphase-normale tags
        if tag.name == "emphase-normale":
            tag.name = "b"
            continue

        # Rename emphase-faible tags
        if tag.name == "emphase-faible":
            tag.name = "em"
            continue

        # Rename note tags
        if tag.name == "note":
            tag.name = "em"

        # # Rename liste tags
        # if tag.name == 'liste':
        #     tag.name = 'ul'
        #     continue

        # # Rename item tags
        # if tag.name == 'item':
        #     tag.name = 'li'
        #     continue

    return soup_copy


def remove_excessive_newlines(text):
    """
    Replaces instances of more than two consecutive newlines with exactly two newlines.

    Parameters:
    text (str): The input text where excessive newlines need to be removed.

    Returns:
    str: The text with excessive newlines replaced by exactly two newlines.
    """
    # Replace instances of more than two consecutive newlines with exactly two newlines
    cleaned_text = re.sub(r"\n{3,}", "\n\n", text)
    return cleaned_text


def md(soup, **options):
    return MarkdownConverter(**options).convert_soup(soup)


# Attention 10613 (id : 1521268) ("xml_null")
# 29281 (definitions pas présentes dans xml)
