import os
import subprocess
from typing import Literal

import pandas as pd
import s3fs

from src.config import Configurable, DefaultFullConfig, FullConfig


def load_dataframe_from_parquet(
    path: str, engine: Literal["auto", "pyarrow", "fastparquet"] = "fastparquet"
) -> pd.DataFrame:
    return pd.read_parquet(path, engine)


@Configurable()
def load_dataframe_from_parquet_using_S3(
    filepath: str,
    engine: Literal["auto", "pyarrow", "fastparquet"] = "fastparquet",
    config: FullConfig = DefaultFullConfig(),
) -> pd.DataFrame:
    fs = s3fs.S3FileSystem(endpoint_url=config.s3_endpoint_url)
    with fs.open(filepath, mode="rb") as file_in:
        return pd.read_parquet(file_in, engine)


def load_chroma_db(s3_path, persist_directory) -> None:
    if not os.path.exists(persist_directory):
        subprocess.run(["mc", "cp", "-r", s3_path, persist_directory])
