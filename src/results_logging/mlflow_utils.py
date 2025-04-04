import shutil
import subprocess
import tempfile
from pathlib import Path, PosixPath

import mlflow
from loguru import logger


def retrieve_unique_collection_id(
    experiment_name: str = "vector_database_building",
    embedding_model: str = "model-org/model-name",
    config: dict = None,
):
    if config is None:
        config = {}

    logger.debug("Checking if collection_name can be retrieved from MLFlow")

    df = mlflow.search_runs(experiment_names=[experiment_name])

    parameters_to_check = ["URL_EMBEDDING_MODEL", "OPENAI_API_BASE", "OPENAI_API_KEY", "QDRANT_URL"]

    eligible_run = (
        df.loc[df["params.embedding_model"] == embedding_model]
        .loc[df["params.OPENAI_API_BASE"] == config.get("OPENAI_API_BASE")]
        .loc[df["params.QDRANT_URL"] == config.get("QDRANT_URL")]
        # .loc[df["params.dirag_only"] == ]
    )

    eligible_run = eligible_run.loc[
        # We remove run when max_pages has been set
        eligible_run["params.max_pages"].isnull()
        # Keep only run on complete dataset
        & eligible_run["params.dataset"]
        != "dirag"
    ]

    n_eligible = eligible_run.shape[0]

    if n_eligible == 0:
        message = (
            f"No eligible run has been found,check your {', '.join(parameters_to_check)} env vars allow to find a model"
        )
        logger.error(message)
        raise ValueError(message)

    if n_eligible > 1:
        logger.debug(f"{n_eligible} runs are consistent with input parameters, taking last one")

    # if multiple run, take the last one
    unique_run_identifier = eligible_run.loc[eligible_run["end_time"] == eligible_run["end_time"].max()][
        "params.QDRANT_COLLECTION_UNIQUE"
    ].iloc[0]

    logger.success(f"Using {unique_run_identifier} collection for evaluation")

    return unique_run_identifier


def mlflow_log_source_files(list_files_to_log=None):
    # Log environment necessary to reproduce the experiment
    current_dir = Path(".")

    if list_files_to_log is None:
        list_files_to_log = list(current_dir.glob("src/**/*.py")) + [PosixPath("run_build_database.py")]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        for file_path in list_files_to_log:
            relative_path = file_path.relative_to(current_dir)
            destination_path = tmp_dir_path / relative_path.parent
            destination_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, destination_path)

        # Generate requirements.txt using pipreqs
        subprocess.run(["pipreqs", str(tmp_dir_path)], check=True)

        # Log all Python files to MLflow artifact
        mlflow.log_artifacts(tmp_dir, artifact_path="environment")
