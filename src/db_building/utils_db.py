import re
from collections import Counter

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import unicodedata

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


def html_tag_finder(xmlstring):
    """
    find all the html type in a xmlstring
    """
    pattern = "<[^<>]+>"
    return Counter(re.findall(pattern, xmlstring))


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


def extract_chunks(row) -> list:
    """
    Extract chunks inside for given page identifier,
    according to the following heuristic: content inside
    of block tags are treated as separate chunks. This is
    an example of a chunking method - its output should
    be studied to see if it is appropriate (chunk lengths, etc.)

    This function is suited for "Chiffres détaillés" pages.

    Args:
        identifier (str): Page identifier.

    Returns:
        List: List of chunks.
    """
    xml_string = str(row["xml_content"])
    identifier = row["id"]

    soup = get_soup(xml_string)

    # Extract chunks
    chunks = []
    for tag in HIGH_LEVEL_TAGS:
        for el in soup(tag):
            for chunk_tag in CHUNK_TAGS:
                while True:
                    tag_chunk = el.find(chunk_tag)
                    if tag_chunk is None:
                        break
                    if tag_chunk.text.strip() != "":
                        chunks.append({"xml": tag_chunk.extract(), "tag": tag, "id": identifier})
                    elif tag_chunk.text.strip() == "":
                        _ = tag_chunk.extract()
            # Here we might be removing some content
            # if CHUNK_TAGS is missing tags
            _ = el.extract()

    # Remaining chunks
    for chunk_tag in CHUNK_TAGS:
        while True:
            chunk = soup.find(chunk_tag)
            if chunk is None:
                break
            if chunk.text.strip() != "":
                chunks.append({"xml": chunk.extract(), "tag": "other", "id": identifier})
            elif chunk.text.strip() == "":
                _ = chunk.extract()

    return chunks


def extract_xml(row) -> tuple:
    """
    Extract content of xml for the given page
    identifier.

    Args:
        identifier (str): Page identifier

    Returns:
        Tuple: xml text chunks, xml tables and links.
    """
    chunks = extract_chunks(row)
    tables = extract_tables(row)
    links = extract_links(row)
    return chunks, tables, links


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


# TODO : Corriger cette fonction, c'est elle qui fait nimp
def paragraph_cleaning(paras, mode=""):
    if mode == "bs":  # read a beautiful soup module
        paras = [p.text.replace("\n", " ") for p in paras]
        paras = [p.replace("\t", "") for p in paras]

    html_tag_re = re.compile(r"<[^>]+>")
    paras = [p.replace("\xa0", "") for p in paras]  # remove \xa0
    paras = [html_tag_re.sub("", p) for p in paras]  # remove html tag
    paras = [re.sub(r" +", " ", s) for s in paras]
    return " ".join(paras)


def theme_parsing(parsed_list: np.array):
    try:
        return " / ".join(list(parsed_list))
    except (ValueError, SyntaxError, TypeError):
        return ""


def extract_paragraphs(table: pd.DataFrame) -> pd.DataFrame:
    """
    extract the paragraphs from the database and associate a relevant url to get access to
    the paragraph on INSEE website. add metadatas associated to textual informations.
    """
    results = {
        "id_origin": [],
        "paragraphs": [],
        "url_source": [],
        "title": [],
        "categories": [],
        "dateDiffusion": [],
        "themes": [],
        "collections": [],
        "libelleAffichageGeo": [],
        "intertitres": [],
        "authors": [],
        "subtitle": [],
    }

    url_bool = "url" in table.columns

    if "xml_content" in table.columns:
        for _, row in tqdm(table.iterrows()):
            try:
                xmlstring = str(row["xml_content"])
                tag = "paragraphe"

                # TODO : Essaier de gérer l'encodage de manière plus fluide + Gérer l'encodage aussi pour intertitre,
                # autheur et sous titre. Idem pour apostrophe
                # title = str(row.titre).replace("\xa0", "")
                soup = BeautifulSoup(xmlstring, "xml")
                paras = soup.find_all(tag)

                # TODO : Faire quelques chose pour rajouter la biblio/citations dans les metadata
                para = paragraph_cleaning(paras, mode="bs")

                # TODO: corriger aussi titre y a rien qui va
                if len(para) > 0:  # filtering to only keep documents with textual informations.
                    results["paragraphs"].append(para.replace("'", "’"))
                    results["id_origin"].append(row.id)
                    results["title"].append(str(row.titre).replace("\xa0", ""))
                    results["categories"].append(row.categorie)
                    results["dateDiffusion"].append(row.dateDiffusion)
                    results["themes"].append(theme_parsing(row.theme))
                    results["collections"].append(row.collection)
                    results["libelleAffichageGeo"].append(row.libelleAffichageGeo)
                    results["intertitres"].append(str(row.xml_intertitre).replace("\xa0", ""))
                    results["authors"].append(str(row.xml_auteurs).replace("\xa0", ""))
                    results["subtitle"].append(str(row.sousTitre).replace("\xa0", "")) if row.sousTitre is not None else results["subtitle"].append(
                        row.sousTitre
                    )

                    if url_bool:
                        results["url_source"].append(row["url"])
            except Exception as e:
                print("issue at this row : ", row)
                print(f"Error : {e}")
        return pd.DataFrame.from_dict(results)


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
            if key.isdigit():
                current_level = current_level[int(key)]
            else:
                current_level = current_level[key]

        if isinstance(current_level, dict):
            # When there is one single dictionnary at the end of the path, we return it as a list
            return [current_level]
        else:
            return current_level
        return current_level

    except (KeyError, TypeError):
        return None


def create_formatted_string(data) -> str:
    formatted_string = []

    for item in data:
        if 'intertitre' in item:
            formatted_string.append(f"\n### {item['intertitre']}\n\n")
        if 'paragraphes' in item and 'paragraphe' in item['paragraphes']:
            for para in item['paragraphes']['paragraphe']:
                formatted_string.append(f"{para}\n")

    return ''.join(formatted_string)


def extract_high_level_tags(element, tags_to_ignore) -> list[str]:
    return [child.name for child in element.find_all(recursive=False) if child.name not in tags_to_ignore]


def recursive_extract(element, tags_to_ignore) -> dict[str, str | dict]:
    # Initialize a dictionary to hold extracted data
    data = {}
    
    # Extract high-level tags from the current element
    high_level_tags = extract_high_level_tags(element, tags_to_ignore)
    
    # Process each high-level tag
    for tag in high_level_tags:
        sub_elements = element.find_all(tag, recursive=False)
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
