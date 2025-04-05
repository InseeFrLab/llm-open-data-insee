import json
import logging
import os

import pandas as pd
import s3fs
from src.data.process import complete_url_builder
from markdownify import markdownify as md

FILES = [
    "applishare_extract",
    "solr_extract",
    "rmes_extract_sources",
    # "applishare_extract_indicateurs",
]
S3_BUCKET = "projet-llm-insee-open-data"

# Logging configuration
logger = logging.getLogger(__name__)


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
    id = get_content(data, "id")
    titre = data.get("titre", "")
    note_historique = md(data.get("noteHistorique", [{}])[0].get("contenu", ""), bullets="-")
    label = get_content(data, "label", 0)
    frequence_collecte = get_content(data, "frequenceCollecte", "label", 0)
    resume = md(data.get("resume", [{}])[0].get("contenu", ""), bullets="-")
    famille = get_content(data, "famille", "label", 0)
    organismes_responsables = get_content(data, "organismesResponsables", 0, "label", 0)
    partenaires = get_content(data, "partenaires", 0, "label", 0)
    # services_collecteurs = get_content(data, "servicesCollecteurs", 0, "label", 0)
    url = f"https://www.insee.fr/fr/metadonnees/source/serie/{id}"
    # author = get_content(data, "autheur", 0)

    parts = [
        f"## {label}",
        f"{titre}",
        f"{url}\n",
        "## Résumé",
        f"{resume}\n",
    ]

    parts.append(f"### Historique\n{note_historique}\n") if note_historique else None
    parts.append(f"### Famille\n{famille}\n") if famille else None
    (parts.append(f"### Organisme responsable\n{organismes_responsables}\n") if organismes_responsables else None)
    (parts.append(f"### Fréquence de collecte des données\n{frequence_collecte}\n") if frequence_collecte else None)
    parts.append(f"### Partenaires\n{partenaires}\n") if partenaires else None
    formatted_page = "\n".join(parts).replace("\\.", ".").replace("\\-", "-")

    return id, label, url, formatted_page  # , resume, author


def process_row(row):
    return extract_rmes_data(json.loads(row))


def main(s3_bucket=S3_BUCKET, list_of_sources=FILES):  # , #config=None):
    fs = s3fs.S3FileSystem(endpoint_url=f"https://{os.getenv('AWS_S3_ENDPOINT')}")

    tables = {
        file: pd.read_parquet(f"s3://{s3_bucket}/data/raw_data/{file}.parquet", filesystem=fs)
        for file in list_of_sources
    }

    for key, table in tables.items():
        logger.info(f"Size of {key} : {len(table)}")

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
    subset_table["dateDiffusion"] = pd.to_datetime(subset_table["dateDiffusion"], format="mixed").dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    subset_table.to_parquet(f"s3://{s3_bucket}/data/raw_data/applishare_solr_joined.parquet", filesystem=fs)

    rmes_sources_content = pd.DataFrame(
        tables["rmes_extract_sources"]["xml_content"].apply(process_row).to_list(),
        columns=["id", "titre", "url", "content"],
    )
    rmes_sources_content.set_index("id").to_parquet(
        f"s3://{s3_bucket}/data/processed_data/rmes_sources_content.parquet",
        filesystem=fs,
    )


if __name__ == "__main__":
    main()
