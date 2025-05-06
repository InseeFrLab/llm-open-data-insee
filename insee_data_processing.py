import os
import json
from loguru import logger

import pandas as pd
import s3fs
from markdownify import markdownify as md

from src.data.url_filling import complete_url_builder
from src.data.parsing import process_xml_rmes_definitions

FILES = [
    "applishare_extract",
    "solr_extract",
    "rmes_extract"
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

# Remove RMES dataset because we have more proper parquet for that
joined_table = joined_table.loc[joined_table['source'] != "rmes"]

# filled_url_x: pages with existing url (retrieved directly from RMes)
# empty_url_x: pages extracted from applishare (url will be reconstructed)
empty_url_x, filled_url_x = (
    joined_table[joined_table["url_x"] == ""],
    joined_table[joined_table["url_x"] != ""]
)

filled_url_x['url'] = filled_url_x['url_x']

# ADD RMES DATA ----------------------------------------

rmes_additional_pages = tables['rmes_extract']
rmes_additional_pages = rmes_additional_pages.rename(
    columns = {"title": "titre", "xml_content": "content"}
)

rmes_data_complete = pd.concat(
    [filled_url_x, rmes_additional_pages]
)


results = {"content": [], "abstract": [], "id": filled_url_x['id'] }

for _, row in filled_url_x.iterrows():
    content, metadata = process_xml_rmes_definitions(row)
    results["content"].append(content)
    results["abstract"].append(metadata)


filled_url_x = (
    filled_url_x
    .drop(columns = "xml_content")
)


# Convertir en DataFrame
filled_url_x = pd.merge(
    filled_url_x, pd.DataFrame(results),
    on = "id"
)

filled_url_x = filled_url_x.rename(
    columns={"content": "xml_content"}
)


# ADD SOLR DATA ------------------------------------------

# 2. Split based on whether url_y is filled or not
url_y_filled, url_y_missing = (
    empty_url_x[empty_url_x["url_y"] != ""],
    empty_url_x[empty_url_x["url_y"] == ""]
)

# FILL url ------------------------------

url_y_missing = complete_url_builder(url_y_missing)

joined_table = pd.concat([
    url_y_missing,
    url_y_filled,
    filled_url_x
])


# ----------------------

subset_table = joined_table.reset_index(drop=True)[
    [
        "id",
        "titre",
        "abstract",
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

print("Done")
