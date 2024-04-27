import logging
import os
import re 
import s3fs

import pandas as pd

#import sys
#sys.path.append("./src")

from config import S3_ENDPOINT_URL
from utils import complete_url_builder





if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create filesystem object
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

    BUCKET = "projet-llm-insee-open-data/data/raw_data" 
    FILE_KEY_S3 = "applishare_extract.parquet"
    FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

    logging.info(FILE_PATH_S3)
    with fs.open(FILE_PATH_S3, mode="rb") as file_in:
        table_app = pd.read_parquet(file_in, engine="fastparquet")

    FILE_KEY_S3 = "solr_extract.parquet"
    FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
    logging.info(FILE_PATH_S3)
    
    with fs.open(FILE_PATH_S3, mode="rb") as file_in:
        table_solr = pd.read_parquet(file_in, engine="fastparquet")

    FILE_KEY_S3 = "rmes_extract_sources.parquet"
    FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
    logging.info(FILE_PATH_S3)
    with fs.open(FILE_PATH_S3, mode="rb") as file_in:
        table_source = pd.read_parquet(file_in, engine="fastparquet")

    FILE_KEY_S3 = "applishare_extract_indicateurs.parquet"
    FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
    logging.info(FILE_PATH_S3)
    with fs.open(FILE_PATH_S3, mode="rb") as file_in:
        table_indic = pd.read_parquet(file_in, engine="fastparquet")

    logging.info("End of loading documents")
    logging.info(f'size table_solr : {len(table_solr)}')
    logging.info(f'size table_app : {len(table_app)}')
    logging.info(f'size table_source : {len(table_source)}')
    logging.info(f'size table_indic : {len(table_indic)}')

    logging.info("Merging Table_solr and table_app operation on id")
    joined_table = table_app.merge(table_solr, how='inner',  on="id")

    logging.info("Rebuilding URL")
    urls_pd_serie = complete_url_builder(joined_table)
    joined_table = joined_table.drop("url", axis=1) #remove url column if exist 
    joined_table.insert(3,"url", urls_pd_serie) 

    logging.info("Storing tables")
    joined_table.to_csv("data_complete.csv", index=False)
