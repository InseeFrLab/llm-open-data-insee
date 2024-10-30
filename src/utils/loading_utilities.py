import os
import subprocess
from collections.abc import Mapping
from typing import Any

import pandas as pd
import s3fs

from src.config import default_config


def load_dataframe_from_parquet(path: str, engine: str = "fastparquet") -> pd.DataFrame:
    return pd.read_parquet(path, engine)


def load_dataframe_from_parquet_using_S3(filepath: str, engine: str = "fastparquet", config: Mapping[str, Any] = default_config) -> pd.DataFrame:
    fs = s3fs.S3FileSystem(endpoint_url=config["s3_endpoint_url"])
    with fs.open(filepath, mode="rb") as file_in:
        return pd.read_parquet(file_in, engine)


def load_chroma_db(s3_path, persist_directory) -> None:
    if not os.path.exists(persist_directory):
        subprocess.run(["mc", "cp", "-r", s3_path, persist_directory])
