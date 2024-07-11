import logging
import re
import unicodedata

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

HIGH_LEVEL_TAGS = [
    "description-generale-variables",
    "documentation-pour-comprendre",
    "dictionnaire-variables",
    "source",
    "definitions",
]

CHUNK_TAGS = [
    "bloc",
    "paragraphe",
    "liste",
]

HIGH_LEVEL_TAGS_TO_IGNORE = [
    "donnees-complementaires",
    "document-imprimable",
    "figure",
    "fichiers-donnees",
    "dictionnaire-variables",
    "tableau",
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
    soup = BeautifulSoup(xml_string, features="xml")
    return soup


def extract_tables(row) -> list:
    """
    Extract tables for a given page identifier.

    Args:
        identifier (str): Page identifier.

    Returns:
        List: Table xml chunks.
    """
    xml_string = str(row["xml_content"])
    identifier = row["id"]

    soup = get_soup(xml_string)

    # Extract tables
    tables = []
    for el in soup.find_all("Tableau"):
        tables.append({"xml": el.extract(), "id": identifier})
    return tables


def extract_links(row) -> list:
    """
    Extract links for a given page identifier.

    Args:
        identifier (str): Page identifier.

    Returns:
        List: Links.
    """
    xml_string = str(row["xml_content"])
    # identifier = row["id"]

    soup = get_soup(xml_string)
    links = []

    # TODO: to complete
    # Find "fichiers-donnees" links
    for el in soup.find_all("fichiers-donnees"):
        title = el.find("titre").get_text()
        uri = el.find("uri").get_text()
        links.append({"title": title, "uri": uri, "type": "data"})

    # External links
    for el in soup.find_all("lien-externe"):
        uri = el.get("url")
        title = el.get_text()
        links.append({"title": title, "uri": uri, "type": "external"})

    return links


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


def find_paths_to_key(nested_dict, target_key, current_path=None, paths_dict=None) -> dict[str, list]:
    if current_path is None:
        current_path = []
    if paths_dict is None:
        paths_dict = {}

    if isinstance(nested_dict, dict):
        for key, value in nested_dict.items():
            new_path = current_path + [key]
            if key == target_key:
                top_level_key = current_path[0] if current_path else key
                if top_level_key not in paths_dict:
                    paths_dict[top_level_key] = []
                paths_dict[top_level_key].append(new_path)
            else:
                find_paths_to_key(value, target_key, new_path, paths_dict)
    elif isinstance(nested_dict, list):
        for index, item in enumerate(nested_dict):
            new_path = current_path + [f"{index}"]
            find_paths_to_key(item, target_key, new_path, paths_dict)

    return paths_dict


def get_value_from_path(nested_dict, path) -> list[dict]:
    current_level = nested_dict
    try:
        for key in path:
            current_level = current_level[int(key)] if key.isdigit() else current_level[key]

        if isinstance(current_level, dict):
            # When there is one single dictionnary at the end of the path, we return it as a list
            return [current_level]
        else:
            return current_level
        return current_level

    except (KeyError, TypeError):
        return None


def create_formatted_string(data: dict) -> str:
    formatted_string = []

    for item in data:
        if "intertitre" in item:
            formatted_string.append(f"\n### {item['intertitre']}\n\n")
        if "paragraphes" in item and "paragraphe" in item["paragraphes"]:
            for para in item["paragraphes"]["paragraphe"]:
                formatted_string.append(f"{para}\n")

    return "".join(formatted_string)


def extract_high_level_tags(element, tags_to_ignore) -> list[str]:
    return [child.name for child in element.find_all(recursive=False) if child.name not in tags_to_ignore]


def recursive_extract(element, tags_to_ignore):
    # Initialize a dictionary to hold extracted data
    data = {}

    # Extract high-level tags from the current element
    high_level_tags = set(extract_high_level_tags(element, tags_to_ignore))

    # Process each high-level tag
    for tag in high_level_tags:
        sub_elements = element.find_all(tag, recursive=False)
        # TODO: Géner ce qu'il se passe à l'interieur des paragraphes. Notamment au niveau des url,
        # italique, gras qui rajoutent des retours à la ligne
        if tag != "paragraphe":
            if len(sub_elements) > 1:
                data[tag] = [recursive_extract(sub_element, tags_to_ignore) for sub_element in sub_elements]
            else:
                sub_element = sub_elements[0]
                if sub_element.find_all(recursive=False):
                    # If the sub-element has children, recurse into it
                    data[tag] = recursive_extract(sub_element, tags_to_ignore)
                else:
                    # If the sub-element is a leaf node, extract its text
                    data[tag] = unicodedata.normalize("NFKD", sub_element.text.strip())
        else:
            data[tag] = [unicodedata.normalize("NFKD", sub_element.text.strip()) for sub_element in sub_elements]
    return data


def format_chapo(data) -> dict:
    # TODO: improve this handler not proper enough
    if len(data["chapo"]) == 1:
        chapo = "\n".join(data["chapo"]["paragraphe"])
    else:
        raise ValueError("Multiple chapo paragraphs found")
    formatted_chapo = f"{chapo}"
    data["chapo"] = formatted_chapo
    return data


def format_external_links(data) -> dict:
    # TODO: improve this handler not proper enough
    if len(data["liens-transverses"]) == 1:
        biblio = "\n".join(data["liens-transverses"]["paragraphe"])
    else:
        raise ValueError("Multiple liens-transverses paragraphs found")
    formatted_biblio = f"{biblio}"
    data["liens-transverses"] = formatted_biblio
    return data


def format_blocs(data: dict) -> dict:
    # TODO : Ici il va falloir prendre en compte les cas ou y a autres choses au dessus des blocs,
    # notamment les titres (e.g pour les sources 22995 et 23000)
    PATHS_WITH_BLOC = find_paths_to_key(data, "bloc")

    for high_level_tag, paths in PATHS_WITH_BLOC.items():
        # Multiple paths found, we merge them all into one
        # usually the cases for "definitions" blocs and encadres blocs
        dict_to_format = [el for path in paths for el in get_value_from_path(data, path)] if len(paths) > 1 else get_value_from_path(data, paths[0])

        formatted_string = create_formatted_string(dict_to_format)
        data[high_level_tag] = formatted_string
    return data


def format_page(data: dict) -> str:
    parts = [
        f"{data.get('titre', '')}",
        f"{data.get('sous-titre', '')}",
        f"{data.get('auteur', '')}\n",
        "## Résumé",
        f"{data.get('chapo', '')}\n",
        f"{data.get('blocs', '')}",
    ]

    # TODO: Onglets à mieux formmater notamment choper les titres
    if "onglets" in data and data["onglets"]:
        parts.append("## Onglets")
        parts.append(data["onglets"])

    # TODO: temporary fix the isinstance check
    if "sources" in data and data["sources"] and isinstance(data["sources"], str):
        parts.append("## Sources")
        parts.append(data["sources"])

    # TODO: temporary fix the isinstance check (when empty dict but not normal)
    if "definitions" in data and data["definitions"] and isinstance(data["definitions"], str):
        parts.append("## Définitions")
        parts.append(data["definitions"])

    if "encadres" in data and data["encadres"]:
        parts.append("## Encadres")
        parts.append(data["encadres"])

    if "liens-transverses" in data and data["liens-transverses"]:
        parts.append("## Références")
        parts.append(data["liens-transverses"])

    formatted_page = "\n".join(parts)
    return formatted_page.strip()


def parse_xmls(data: pd.DataFrame, id: str = "id", xml_column: str = "xml_content") -> pd.DataFrame:
    parsed_pages = {"id": [], "content": []}

    for i, row in data.iterrows():
        page_id = row[id]
        logging.info(f"Processing page {page_id} -- {i}/{len(data)}")

        if not row[xml_column]:
            # When xml_content is empty, we skip the page
            # TODO: (to be changed, we should extract the xml in the first place)
            continue

        soup = get_soup(row[xml_column])
        root = soup.find()

        # Extract data from the XML
        result = recursive_extract(root, HIGH_LEVEL_TAGS_TO_IGNORE)

        # Format chapo if it exists and is not empty
        result = format_chapo(result) if "chapo" in result and result["chapo"] else result

        # Format all high level tags that contains bloc tags (corps de texte, encadres, définitions etc.)
        result = format_blocs(result)

        # Format external links if it exists and is not empty
        result = format_external_links(result) if "liens-transverses" in result and result["liens-transverses"] else result

        formatted_page = format_page(result)
        parsed_pages["id"].append(page_id)
        parsed_pages["content"].append(formatted_page)

    return pd.DataFrame(parsed_pages).set_index("id")


# Attention 10613 (id : 1521268) pas bien récupéré, xml vide
# 6699
# 6700
# 6701
# 6702
# 6703
# 6704
# 6705
# 10613 ("xml_null")
# 29281 (definitions pas présentes dans xml)
# 31083 (long à vérifier)
# 39570 (long à vérifier)
