import logging
import os
import pandas as pd
#import s3fs

def load_dataframe_from_parquet(path: str, engine: str = 'fastparquet'):
    return pd.read_parquet(path, engine)

#def load_dataframe_from_parquet_using_S3(S3_FILE_PATH, engine: str = 'fastparquet'):
#    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})
#    with fs.open(S3_FILE_PATH, mode="rb") as file_in:
#        return pd.read_parquet(file_in, engine)
