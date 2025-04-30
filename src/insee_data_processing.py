import os
import json
from loguru import logger

import pandas as pd
import s3fs
from markdownify import markdownify as md

from src.data.process import complete_url_builder

FILES = [
    "applishare_extract",
    "solr_extract",
]
S3_BUCKET = "projet-llm-insee-open-data"


fs = s3fs.S3FileSystem(endpoint_url=f"https://{os.getenv('AWS_S3_ENDPOINT')}")


# LECTURE DES TABLES ---------------------------

tables = {
    file: pd.read_parquet(f"s3://{S3_BUCKET}/data/raw_data/{file}.parquet", filesystem=fs)
    for file in FILES
}

for key, table in tables.items():
    logger.info(f"Size of {key} : {len(table)}")

joined_table = tables["applishare_extract"].merge(
    tables["solr_extract"], how="left", on="id"
)

# FILL url ------------------------------

empty_url_x, filled_url_x = (
    joined_table[joined_table["url_x"] == ""],
    joined_table[joined_table["url_x"] != ""]
)

# 2. Split based on whether url_y is filled or not
url_y_filled, url_y_missing = (
    empty_url_x[empty_url_x["url_y"] != ""],
    empty_url_x[empty_url_x["url_y"] == ""]
)

url_y_missing = complete_url_builder(url_y_missing)
filled_url_x['url'] = filled_url_x['url_x']

joined_table = pd.concat([
    url_y_missing,
    url_y_filled,
    filled_url_x
])


# ----------------------

joined_table["theme"] = [x[0] if x is not None else x for x in joined_table["theme"]]


subset_table = joined_table.reset_index(drop=True)[
    [
        "id",
        "titre",
        "categorie",
        "url",
        "dateDiffusion",
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
subset_table.to_parquet(f"s3://{S3_BUCKET}/data/raw_data/applishare_solr_joined_new.parquet", filesystem=fs)

