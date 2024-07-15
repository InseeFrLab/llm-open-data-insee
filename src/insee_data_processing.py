import json
import logging

import pandas as pd
import s3fs
from config import S3_BUCKET, S3_ENDPOINT_URL
from db_building.utils_db import complete_url_builder
from markdownify import markdownify as md

FILES = [
    "applishare_extract",
    "solr_extract",
    "rmes_extract_sources",
    # "applishare_extract_indicateurs",
]


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


def extract_rmes_data(data: dict):
    """
    Extracts and processes specific fields from the input data dictionary.

    Args:
        data (dict): The input data dictionary containing various fields.

    Returns:
        dict: A dictionary containing the processed fields.
    """

    titre = data.get("titre", "")
    note_historique = md(data.get("noteHistorique", [{}])[0].get("contenu", ""), bullets="-")
    label = get_content(data, "label", 0)
    frequence_collecte = get_content(data, "frequenceCollecte", "label", 0)
    resume = md(data.get("resume", [{}])[0].get("contenu", ""), bullets="-")
    famille = get_content(data, "famille", "label", 0)
    organismes_responsables = get_content(data, "organismesResponsables", 0, "label", 0)
    partenaires = get_content(data, "partenaires", 0, "label", 0)
    # services_collecteurs = get_content(data, "servicesCollecteurs", 0, "label", 0)
    url = f"https://www.insee.fr/fr/metadonnees/source/serie/{get_content(data, "id")}"

    parts = [
        f"## {label}",
        f"{titre}",
        f"{url}\n",
        "## Résumé",
        f"{resume}\n",
    ]

    parts.append(f"### Historique\n{note_historique}\n") if note_historique else None
    parts.append(f"### Famille\n{famille}\n") if famille else None
    parts.append(f"### Organisme responsable\n{organismes_responsables}\n") if organismes_responsables else None
    parts.append(f"### Fréquence de collecte des données\n{frequence_collecte}\n") if frequence_collecte else None
    parts.append(f"### Partenaires\n{partenaires}\n") if partenaires else None
    formatted_page = "\n".join(parts).replace("\\.", ".").replace("\\-", "-")

    return url, formatted_page


def process_row(row):
    return extract_rmes_data(json.loads(row))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    tables = {file: pd.read_parquet(f"s3://{S3_BUCKET}/data/raw_data/{file}.parquet", filesystem=fs) for file in FILES}

    for key, table in tables.items():
        logging.info(f"Size of {key} : {len(table)}")

    joined_table = tables["applishare_extract"].merge(tables["solr_extract"], how="inner", on="id")
    joined_table["url"] = complete_url_builder(joined_table)
    joined_table["theme"] = [x[0] if x is not None else x for x in joined_table["theme"]]
    subset_table = joined_table.reset_index(drop=True)[
        [
            "id",
            "titre",
            "categorie",
            "url",
            "dateDiffusion",
            "theme",
            "collection",
            "libelleAffichageGeo",
            "xml_intertitre",
            "xml_auteurs",
            "sousTitre",
            "xml_content",
        ]
    ]
    subset_table["dateDiffusion"] = pd.to_datetime(subset_table["dateDiffusion"], format="mixed").dt.strftime("%Y-%m-%d %H:%M")
    subset_table.to_parquet(f"s3://{S3_BUCKET}/data/raw_data/applishare_solr_joined.parquet", filesystem=fs)

    rmes_sources_content = pd.DataFrame(tables["rmes_extract_sources"]["xml_content"].apply(process_row).to_list(), columns=["url", "content"])
    rmes_sources_content.to_parquet(f"s3://{S3_BUCKET}/data/processed_data/rmes_sources_content.parquet", filesystem=fs)


if __name__ == "__main__":
    main()
