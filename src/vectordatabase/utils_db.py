import re

# import logging
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag
from loguru import logger
from markdownify import MarkdownConverter, markdownify

# logger = logging.getLogger(__name__)

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
# TODO: make it a parameter of the function (want tableau to be a parameter)


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


def prepend_text_to_tag(tag, text):
    """
    Prepends the given text to the content of the specified tag.

    Parameters:
    tag (Tag): The BeautifulSoup tag object to which the text will be prepended.
    text (str): The text to prepend to the tag's content.

    """
    if tag.get_text() is not None:
        tag.insert(0, text)
    else:
        tag.string = text


def parse_xmls(data: pd.DataFrame, id: str = "id", xml_column: str = "xml_content") -> pd.DataFrame:
    """
    Parses XML content from a DataFrame, extracts data, formats it,
    and returns a new DataFrame with the formatted content.

    Args:
    - data (pd.DataFrame): The input DataFrame containing XML data.
    - id (str, optional): The column name for the unique identifier of each row. Defaults to "id".
    - xml_column (str, optional): The column name containing the XML content.
        Defaults to "xml_content".

    Returns:
    - pd.DataFrame: A DataFrame with 'id' as the index and formatted content in the 'content' column
    """

    data = data.reset_index(names="index")
    parsed_pages: dict[str, list] = {"id": [], "content": [], "abstract": []}

    logstep = 1 + (len(data) // 10)
    for i, row in data.iterrows():
        page_id = row["id"]
        page_number = row["index"]
        if i % logstep == 0:
            logger.info(f"Parsing XML from page {page_number} -- {i}/{len(data)} ({100 * i / len(data):.2f}%)")

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

        h2_tag = soup.find("h2")
        abstract = markdownify("\n".join([str(content) for content in h2_tag.contents])) if h2_tag is not None else ""

        parsed_pages["id"].append(page_id)
        parsed_pages["content"].append(remove_excessive_newlines(parsed_page))
        parsed_pages["abstract"].append(abstract.replace("Résumé :\n\n", ""))

    data_as_md = pd.DataFrame(parsed_pages).set_index("id")

    return data_as_md


def format_tags(soup: Tag, tags_to_ignore: list[str]) -> Tag:
    """
    Formats the tags in the provided BeautifulSoup object according to specific HTML rules.

    Parameters:
    ----------
    soup : Tag
        The BeautifulSoup tag object to format.
    tags_to_ignore : list[str]
        A list of tag names to ignore and remove.

    Returns:
    --------
    Tag
        The formatted BeautifulSoup tag object.
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
        # We need to do that because some content (e.g titles, sources...) are in the figure tag
        # and not in the children tags (graphique, tableau...)
        # Maybe we still want to keep it ?
        if any(remove_figure) and tag.name == "figure":
            if all(
                tag.find(child_tag) is None
                for child_tag, remove in zip(TAGS_FIGURE_CHILDREN, remove_figure, strict=False)
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
            tag.name = f"h{int(tag.get('niveau')) + 2}" if tag.get("niveau") is not None else "h3"
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
