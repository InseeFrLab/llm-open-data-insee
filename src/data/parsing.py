import unicodedata
import re

# import logging
import pandas as pd
from bs4 import BeautifulSoup, NavigableString
from bs4.element import Tag
from loguru import logger
from markdownify import MarkdownConverter, markdownify

# logger = logging.getLogger(__name__)

TABLE_ADHOC_SEPARATOR = "\n\n-----------------\n\n"
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


# RAW DATA PARSING -------------------------------------

def get_content(data, *keys):
    """
    Safely retrieves nested content from a dictionary or list using a sequence of keys and indices.

    Args:
        data (dict or list): The dictionary or list to retrieve content from.
        *keys: A sequence of keys and/or indices to navigate the nested structure.

    Returns:
        str: The retrieved content or an empty string if any key or index is missing.
    """
    for key in keys:
        if isinstance(data, list):
            if not (isinstance(key, int) and 0 <= key < len(data)):
                return ""
            data = data[key]
        elif isinstance(data, dict):
            data = data.get(key, {})
        else:
            return ""
    return data if isinstance(data, str) else data.get("contenu", "")


def process_xml_rmes_definitions(row):
    """
    Restructure le XML renvoyé par RMes API (metadata)

    Parameters:
        row (pd.Series): Ligne du DataFrame contenant au moins 'xml_content' et 'titre'.

    Returns:
        tuple[str, str]: Contenu formaté (markdown), métadonnées brutes.
    """
    content = row["xml_content"]
    title = row["titre"]

    soup = BeautifulSoup(content)

    # Supprime <definitionsliees> pour pas doublonner avec les définitions qu'on a déjà
    tag = soup.find("definitionsliees")
    if tag:
        tag.decompose()

    # Extraction de blocs de texte
    chapo = extract_text_joined(soup, "chapo")
    remarque = extract_text_joined(soup, "remarque")
    metadata = extract_text_joined(soup, "chapometadonnees")

    # Traitement des synonymes
    synonymes_tags = soup.find_all("synonymes")
    synonymes_list = [tag.get_text(strip=True) for tag in synonymes_tags if tag.get_text(strip=True)]

    synonymes = f"### Concepts synonymes: {'; '.join(synonymes_list)}" if synonymes_list else ""

    # Contenu final
    content = f"<h1>{title}</h1>\n\n<h2> Définition</h2>\n{chapo}\n\n<h2>Remarque</h2>\n{remarque}\n\n{synonymes}"

    return content, metadata


# XML PARSING -----------------------------------------------------

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


def prepend_text_to_tag(tag: Tag, text: str):
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


def extract_tables_from_page(soup: Tag, drop_figures: bool = True):
    tables = soup.find_all("table")

    container = soup.new_tag("div")

    # Remove table from the page (but keep them elsewhere)
    for idx, table in enumerate(tables):
        table.extract()
        container.append(table)
        if idx != len(tables) - 1:  # Don't add separator after the last table
            separator = NavigableString(TABLE_ADHOC_SEPARATOR)
            container.append(separator)

    if drop_figures is True:
        figures = soup.find_all("figure")
        # Dropping figures
        for figure in figures:
            figure.decompose()

    # Remove all figures
    return soup, container


def clean_text(text):
    """
    Compose all cleaning steps to normalize, remove control characters,
    collapse whitespace, and fix excessive newlines.
    """
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = collapse_whitespace(text)
    text = remove_excessive_newlines(text)
    return text.strip()


def extract_text_joined(soup, tag_name, separator="\n"):
    """
    Extracts and joins text from all occurrences of a given tag.

    Parameters:
        soup (BeautifulSoup): The parsed HTML/XML soup.
        tag_name (str): Name of the tag to extract.
        separator (str): String to join text blocks (default: newline).

    Returns:
        str: Combined text content of all found tags, separated by `separator`.
    """
    return separator.join(tag.get_text(strip=True) for tag in soup.find_all(tag_name))



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


def normalize_unicode(text):
    """Normalize Unicode characters using NFKC."""
    return unicodedata.normalize("NFKC", text)

def remove_control_chars(text):
    """Remove non-printable characters, except newlines."""
    return ''.join(ch for ch in text if ch.isprintable() or ch == '\n')

def collapse_whitespace(text):
    """Replace all non-newline whitespace (incl. \xa0, tabs) with a single space."""
    return re.sub(r"[^\S\n]+", " ", text)

def remove_excessive_newlines(text):
    """Replace 3+ consecutive newlines with exactly two newlines."""
    return re.sub(r"\n{3,}", "\n\n", text)


def parse_xmls(
    data: pd.DataFrame,
    id: str = "id",
    xml_column: str = "xml_content",
    rename_tags: bool = True,
    create_abstract: bool = True
    ) -> pd.DataFrame:
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

    args_markdown_converter = {"escape_misc": False, "escape_asterisks": False, "bullets": "-", "heading_style": "ATX"}

    data = data.reset_index(names="index")
    parsed_pages: dict[str, list] = {"id": [], "content": [], "abstract": [], "tables": []}

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
        if rename_tags is True:
            soup = format_tags(get_soup(row[xml_column]), TAGS_TO_IGNORE)
        else:
            soup = BeautifulSoup(row[xml_column])

        # Isolating tables
        soup, tables = extract_tables_from_page(soup, drop_figures=True)

        # Transform the xml content into markdown
        parsed_page = md(
            soup,
            **args_markdown_converter,
        )

        parsed_tables = md(tables, **args_markdown_converter)

        h2_tag = soup.find("h2")
        if create_abstract is True:
            abstract = markdownify("\n".join([str(content) for content in h2_tag.contents])) if h2_tag is not None else ""
        else:
            abstract = row['abstract']

        parsed_pages["id"].append(page_id)
        parsed_pages["content"].append(clean_text(parsed_page))
        parsed_pages["abstract"].append(abstract.replace("Résumé :\n\n", ""))
        if parsed_tables:
            parsed_pages["tables"].append(clean_text(parsed_tables))
        else:
            parsed_pages["tables"].append("")

    data_as_md = pd.DataFrame(parsed_pages).set_index("id")

    return data_as_md


def parse_documents(data: pd.DataFrame, xml_column: str = "xml_content") -> pd.DataFrame:

    logger.info("Parsing XML content")

    main_pages = data.loc[~data["id"].str.match(r"^[A-Za-z]")]
    special_pages = data.loc[data["id"].str.match(r"^[A-Za-z]")]

    logger.info("Parsing most pages")
    parsed_pages = parse_xmls(main_pages, xml_column=xml_column)

    logger.info("Parsing pages retrieved from API")
    parsed_pages_special = parse_xmls(
        special_pages, xml_column=xml_column,
        rename_tags=False, create_abstract=False
    )

    parsed_pages = pd.concat([
            parsed_pages, parsed_pages_special
    ])

        # Merge parsed XML data with the original DataFrame
    df = (
            data.drop(columns="abstract").set_index("id")
            .merge(parsed_pages, left_index=True, right_index=True)
            .drop(columns=["xml_content"], errors="ignore")  # Drop only if exists
    )

    df = df.loc[
        :,
        [
            "titre",
            "categorie",
            "url",
            "dateDiffusion",
            "collection",
            "libelleAffichageGeo",
            "content",
            "abstract",
            "tables",
        ],
    ]

    df = df.fillna(value="")

    return df


# CACHING PARSED DOCUMENTS ----------------------------


def md(soup, **options):
    return MarkdownConverter(**options).convert_soup(soup)
