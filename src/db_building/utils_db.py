import re
from collections import Counter

import numpy as np
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
