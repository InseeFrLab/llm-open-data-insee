import logging

import pandas as pd
import s3fs
from config import S3_BUCKET, S3_ENDPOINT_URL
from db_building.utils_db import complete_url_builder

FILES = [
    "applishare_extract",
    "solr_extract",
    # "rmes_extract_sources",
    # "applishare_extract_indicateurs",
]


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


if __name__ == "__main__":
    main()
