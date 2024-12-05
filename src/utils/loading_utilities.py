from typing import Literal

import pandas as pd


def load_dataframe_from_parquet(
    path: str, engine: Literal["auto", "pyarrow", "fastparquet"] = "fastparquet"
) -> pd.DataFrame:
    return pd.read_parquet(path, engine)
