###### data 
import pandas as pd
import os
import s3fs
import re 

###### custom files
from utils import complete_url_builder

if __name__ == '__main__':

    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

    BUCKET = "projet-llm-insee-open-data/data/raw_data" 
    FILE_KEY_S3 = "applishare_extract.parquet"
    FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

    print(FILE_PATH_S3)
    with fs.open(FILE_PATH_S3, mode="rb") as file_in:
        table_app = pd.read_parquet(file_in, engine="fastparquet")

    FILE_KEY_S3 = "solr_extract.parquet"
    FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
    print(FILE_PATH_S3)
    with fs.open(FILE_PATH_S3, mode="rb") as file_in:
        table_solr = pd.read_parquet(file_in, engine="fastparquet")

    FILE_KEY_S3 = "rmes_extract_sources.parquet"
    FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
    print(FILE_PATH_S3)
    with fs.open(FILE_PATH_S3, mode="rb") as file_in:
        table_source = pd.read_parquet(file_in, engine="fastparquet")

    FILE_KEY_S3 = "applishare_extract_indicateurs.parquet"
    FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
    print(FILE_PATH_S3)
    with fs.open(FILE_PATH_S3, mode="rb") as file_in:
        table_indic = pd.read_parquet(file_in, engine="fastparquet")

    print("End of loading documents")
    print(f'size table_solr : {len(table_solr)}')
    print(f'size table_app : {len(table_app)}')
    print(f'size table_source : {len(table_source)}')
    print(f'size table_indic : {len(table_indic)}')

    print("Merging Table_solr and table_app operation on id")
    joined_table = table_app.merge(table_solr, how='inner',  on="id")

    print("Rebuilding URL")
    urls_pd_serie = complete_url_builder(joined_table)
    joined_table = joined_table.drop("url", axis=1) #remove url column if exist 
    joined_table.insert(3,"url", urls_pd_serie) 

    print("Storing tables")
    joined_table.to_csv("src/data/data_complete.csv", index=False)
