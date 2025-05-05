import re

import pandas as pd
import s3fs
from loguru import logger

DEFAULT_WEB4G_LOCATION = "s3://projet-llm-insee-open-data/data/raw_data/applishare_extract.parquet"
DEFAULT_RMES_LOCATION = "s3://projet-llm-insee-open-data/data/processed_data/rmes_sources_content.parquet"
DEFAULT_FS = s3fs.S3FileSystem(endpoint_url="https://minio.lab.sspcloud.fr")


def constructor_corpus(field: str = "complete", **kwargs):
    if field not in {"complete", "dirag"}:
        raise ValueError("Only 'complete' and 'dirag' values for 'field' argument are accepted")
    constructor = _constructor_data_complete if field == "complete" else _constructor_data_dirag
    return constructor(**kwargs)


def _constructor_data_complete(
    web4g_path_uri: str = DEFAULT_WEB4G_LOCATION,
    rmes_path_uri: str = DEFAULT_RMES_LOCATION,
    fs: s3fs.S3FileSystem = DEFAULT_FS,
    **kwargs,
):
    if kwargs.get("search_cols"):
        logger.debug("'search_cols' provided but ignored since we don't restrict our corpus")

    logger.info("Whole insee.fr website will be processed")
    logger.info(f"Reading dataset ({web4g_path_uri} and {rmes_path_uri} locations)")
    data = pd.read_parquet(web4g_path_uri, filesystem=fs)

    if kwargs.get("with_rmes", False) is False:
        return data

    data_rmes = (
        pd.read_parquet(rmes_path_uri, filesystem=fs).rename({"content": "xml_content"}, axis="columns")
        # rename column to ensure relevant info in same column in both df
    )

    df = pd.concat([data, data_rmes], axis=0)
    return df


def _constructor_data_dirag(
    web4g_path_uri: str = DEFAULT_WEB4G_LOCATION,
    fs: s3fs.S3FileSystem = DEFAULT_FS,
    search_cols: str | list[str] = None,
) -> pd.DataFrame:
    """
    Reads a Parquet file from S3, filters the data based on a regex pattern,
    and returns the filtered DataFrame.

    Args:
        s3_path (str): The S3 path to the Parquet file.
        fs (s3fs.S3FileSystem): An instance of S3FileSystem to interact with S3.
        search_cols (Union[str, List[str]], optional): Columns to search for the regex pattern.
            When None, set to ["titre", "libelleAffichageGeo", "xml_intertitre"].

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows matching the regex pattern.
    """

    if search_cols is None:
        search_cols = ["titre", "libelleAffichageGeo", "xml_intertitre"]

    regex_identification_dirag = r"(antilla|antille|martiniq|guadelou|guyan)"

    # Read Parquet data from S3
    logger.info("Restricting insee.fr to DIRAG corpus")
    logger.info(f"Reading dataset ({web4g_path_uri} location)")
    donnees_site_insee = pd.read_parquet(web4g_path_uri, engine="pyarrow", filesystem=fs)

    # Define the regex pattern (case-insensitive)
    pattern_antilles = re.compile(regex_identification_dirag, re.IGNORECASE)

    # Apply regex filtering across the selected columns
    mask = (
        donnees_site_insee.loc[:, search_cols]
        .apply(lambda col: col.str.contains(pattern_antilles, na=False))
        .any(axis=1)
    )

    return donnees_site_insee.loc[mask]
