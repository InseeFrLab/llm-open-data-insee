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


