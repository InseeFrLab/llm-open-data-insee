import pandas as pd
import s3fs

from config import S3_ENDPOINT_URL


def load_dataframe_from_parquet(path: str, engine: str = "fastparquet"):
    return pd.read_parquet(path, engine)


def load_dataframe_from_parquet_using_S3(filepath: str, engine: str = "fastparquet"):
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})
    with fs.open(filepath, mode="rb") as file_in:
        return pd.read_parquet(file_in, engine)
