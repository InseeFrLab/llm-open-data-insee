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

empty_url_x = joined_table[joined_table["url_x"] == ""]

# 2. Split based on whether url_y is filled or not
url_y_filled = empty_url_x[empty_url_x["url_y"] != ""]
url_y_missing = empty_url_x[empty_url_x["url_y"] == ""]

url_y_filled.sum()

# ----------------------

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
