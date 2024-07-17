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


def extract_text_tag(xmlstring, tag="paragraphe"):
    """
    Extract the elements between xml tag , note that the html and xml file associated to the same page are different,
    xml files do not have any attribute like paragraphe-chapeau, ...
    """
    # tag paragraphe without attribute
    pattern = "<" + tag + ">(.*?)</" + tag + ">"
    return re.findall(pattern, xmlstring)


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


def url_builder(row: pd.Series):
    category = row.categorie

    if category is None:
        return None

    i = row.id

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

    if category in set(dict_url.keys()):
        base_url = "https://www.insee.fr/fr/"

        section = dict_url[category]

        if section is None or len(section) == 0:
            return None

        url = base_url + section + "/" + i

        return url
    else:
        return None


def url_builder_metadata(row: pd.Series):
    """
    rebuild valid URL for metadata
    base url looks like : https://www.insee.fr/fr/metadonnees/
    """

    base_url = "https://www.insee.fr/fr/metadonnees/"
    pattern_source = r"^s"
    pattern_indicator = r"^p"

    row_id = row.id

    if re.match(pattern_source, row_id):
        return base_url + f"source/serie/{row_id}"

    if re.match(pattern_indicator, row_id):
        return base_url + f"source/indicateur/{row_id}/description"

    return None


def complete_url_builder(table):
    urls = []
    for _, row in tqdm(table.iterrows()):
        url = url_builder(row)
        if url is None:
            url = url_builder_metadata(row)

        urls.append(url)  # add URL or None

    return pd.Series(urls)


def parse_xmls(data: pd.DataFrame, id: str = "id", xml_column: str = "xml_content") -> pd.DataFrame:
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

        # Extract the xml content and making it hmtl compliant
        soup = format_tags(get_soup(row[xml_column]), TAGS_TO_IGNORE)

        soup.find("liste")

        # Transform the xml content into markdown
        parsed_page = md(soup, escape_misc=False, escape_asterisks=False, bullets="-", heading_style="ATX")

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
    soup_copy = soup
    TAGS_FIGURE_CHILDREN = ["graphique", "tableau"]
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
            if any([tag.find(TAGS_FIGURE_CHILDREN[i]) is not None for i, value in enumerate(remove_figure) if value is False]):
                # This make sure there is no children tags to keep in the figure tag
                continue
            else:
                for i, rm_fig in enumerate(remove_figure):
                    tag.decompose() if rm_fig and tag.find(TAGS_FIGURE_CHILDREN[i]) else None

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
        if tag.name == "definitions":
            prepend_text_to_tag(tag, "Définitions : ")
            tag.name = "h2"
            continue

        if tag.name == "liens-transverses":
            txt = tag.get("titre") if tag.get("titre") is not None else "Références"
            prepend_text_to_tag(tag, txt)
            tag.name = "h2"
            continue

        # Rename intertitre tags
        if tag.name == "intertitre":
            tag.name = f"h{int(tag.get("niveau")) + 2}" if tag.get("niveau") is not None else "h3"
            continue

        # Rename avertissement tags
        if tag.name == "avertissement" and len(list(tag.parents)) == 2:
            tag.name = "h2"
            continue

        # Rename lignes tags
        if tag.name == "lignes":
            if tag.has_attr("type") and tag["type"] == "entete":
                tag.name = "thead"
            elif tag.has_attr("type") and tag["type"] == "donnees":
                tag.name = "tbody"
            continue

        # Rename ligne tags
        if tag.name == "ligne":
            tag.name = "tr"
            continue

        # Rename cellule entete tags
        if tag.name == "cellule":
            if tag.has_attr("entete") and tag["entete"] == "colonne":
                tag.name = "th"
            # elif tag.has_attr('entete') and tag['entete'] == 'ligne':
            #     tag.name = 'td'
            else:
                tag.name = "td"
            continue

        # Rename paragraphe tags
        if tag.name == "paragraphe":
            tag.name = "p"
            continue

        # Rename tableau tags
        if tag.name == "tableau":
            tag.name = "table"
            continue

        # Rename lien-externe tags
        if tag.name == "lien-externe":
            tag.name = "a"
            tag["href"] = tag.get("url")
            if tag.get("url") is not None:
                del tag["url"]
            continue

        # # Rename liste tags
        # if tag.name == 'liste':
        #     tag.name = 'ul'
        #     continue

        # # Rename item tags
        # if tag.name == 'item':
        #     tag.name = 'li'
        #     continue

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

    return soup_copy


def remove_excessive_newlines(text):
    # Replace instances of more than three consecutive newlines with exactly three newlines
    cleaned_text = re.sub(r"\n{3,}", "\n\n", text)
    return cleaned_text


def prepend_text_to_tag(tag, text):
    if tag.string:
        tag.string.insert_before(text)
    else:
        tag.insert(0, text)


def md(soup, **options):
    return MarkdownConverter(**options).convert_soup(soup)


# Attention 10613 (id : 1521268) ("xml_null")
# 29281 (definitions pas présentes dans xml)
