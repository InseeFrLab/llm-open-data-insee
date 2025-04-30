import re
from tqdm import tqdm

def url_builder(row: pd.Series) -> str | None:
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


def complete_url_builder(table: pd.DataFrame) -> pd.Series:
    """
    Builds a complete list of URLs for each row in the given DataFrame.

    If the URL cannot be constructed using `url_builder`, it tries `url_builder_metadata`.

    Parameters:
    table (pd.DataFrame): A pandas DataFrame containing rows with necessary fields.

    Returns:
    pd.Series: A pandas Series containing the constructed URLs or None for each row.
    """

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

    base_url = "https://www.insee.fr/fr"

    table['url'] = (
        base_url + "/" +
        table["categorie"].map(dict_url) + "/" +
        table["id"]
    )

    return table
