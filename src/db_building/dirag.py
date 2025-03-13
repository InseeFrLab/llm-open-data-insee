import logging
from typing import Union, List
import re

import pandas as pd
import s3fs

logger = logging.getLogger(__name__)


def constructor_data_dirag(
    s3_path: str,
    fs: s3fs.S3FileSystem,
    search_cols: Union[str, List[str]] = ["titre", "libelleAffichageGeo", "xml_intertitre"],
) -> pd.DataFrame:
    """
    Reads a Parquet file from S3, filters the data based on a regex pattern,
    and returns the filtered DataFrame.

    Args:
        s3_path (str): The S3 path to the Parquet file.
        fs (s3fs.S3FileSystem): An instance of S3FileSystem to interact with S3.
        search_cols (Union[str, List[str]], optional): Columns to search for the regex pattern.
            Defaults to ["titre", "libelleAffichageGeo", "xml_intertitre"].

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows matching the regex pattern.
    """

    regex_identification_dirag = r"(antilla|antille|martiniq|guadelou|guyan)"

    # Read Parquet data from S3
    donnees_site_insee = pd.read_parquet(s3_path, engine="pyarrow", filesystem=fs)

    # Define the regex pattern (case-insensitive)
    pattern_antilles = re.compile(regex_identification_dirag, re.IGNORECASE)

    # Apply regex filtering across the selected columns
    mask = (
        donnees_site_insee.loc[:, search_cols]
        .apply(lambda col: col.str.contains(pattern_antilles, na=False))
        .any(axis=1)
    )

    return donnees_site_insee.loc[mask]
