import argparse
import ast
import logging
import os
import sys
from configparser import ConfigParser, ExtendedInterpolation

import mlflow

# PARSER FOR USER LEVEL ARGUMENTS --------------------------------

argparser = argparse.ArgumentParser(description="Chroma building parameters")
argparser.add_argument(
    "--experiment_name",
    type=str,
    help="""
    Name of the experiment.
    """,
)
argparser.add_argument(
    "--data_raw_s3_path",
    type=str,
    help="""
    Path to the raw data.
    Default to data/raw_data/applishare_solr_joined.parquet
    """,
)
argparser.add_argument(
    "--collection_name",
    type=str,
    help="""
    Collection name.
    Default to insee_data
    """,
)
argparser.add_argument(
    "--markdown_split",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="""
    Should we use a markdown split ?
    --markdown_split yields True and --no-markdown_split yields False
    """,
)
argparser.add_argument(
    "--use_tokenizer_to_chunk",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="""
    Should we use the tokenizer of the embedding model to chunk ?
    --use_tokenizer_to_chunk yields True and --no-use_tokenizer_to_chunk yields False
    """,
)
argparser.add_argument(
    "--separators",
    help="List separators to split the text",
)
argparser.add_argument(
    "--embedding_model",
    type=str,
    dest="emb_model",
    help="""
    Embedding model.
    Should be a huggingface model.
    Defaults to OrdalieTech/Solon-embeddings-large-0.1
    """,
)
argparser.add_argument(
    "--max_pages",
    type=int,
    help="""
    Maximum number of pages to use for the vector database.
    """,
)
argparser.add_argument(
    "--chunk_size",
    type=str,
    help="""
    Chunk size
    """,
)
argparser.add_argument(
    "--chunk_overlap",
    type=str,
    default=None,
    help="""
    Chunk overlap
    """,
)
argparser.add_argument(
    "--embedding_device",
    type=str,
    dest="emb_device",
    help="""
    Embedding device
    """,
)
argparser.add_argument(
    "--force_rebuild",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="""
    Should we reuse previously constructed database or rebuild
    """,
)
argparser.add_argument(
    "--config_file",
    type=str,
    metavar="INIFILE",
    help="""
    Specify a config file from which parameters can be read
    """,
)
argparser.add_argument(
    "--config_mlflow",
    type=str,
    metavar="RUNID",
    dest="mlflow_run_id",
    help="""
    Load configuration from a previous mlflow run.
    Other flags will override this configuration
    but neither will .ini config files nor environment variables.
    """,
)
argparser.add_argument(
    "--export_config",
    default=False,
    action="store_true",
    help="""
    Prints the full configuration to stdout and immediately terminates
    """,
)
argparser.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    default=False,
    action="store_true",
    help="""
    Set logging level to DEBUG
    """,
)
argparser.add_argument(
    "-l",
    "--log",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    nargs="?",
    dest="loggingLevel",
    const="INFO",
    default="INFO",
    type=str.upper,
    help="""
    Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """,
)


class PostProcessedConfigParser(ConfigParser):
    """
    ConfigParser with registered internal post-processors that are run on templated options.
    Only the `get` method is modified. Sections of this ConfigParser should work fine as they
    rely on this method to access options.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._processors = {}

    def setProcessor(self, option, proc):
        self._processors[self.optionxform(option)] = proc

    def setProcessors(self, procs):
        for opt, proc in procs.items():
            self.setProcessor(opt, proc)

    def get(self, section, option, **kwargs):
        tpl_opt = super().get(section, option, **kwargs)
        proc = self._processors.get(self.optionxform(option))
        return proc(tpl_opt) if proc and tpl_opt is not None else tpl_opt

    def update_dict(self, d, section="DEFAULT"):
        """Updates the already loaded parameters with (not None) values from given dict"""
        self.read_dict({section: {k: str(v) for k, v in d.items() if v is not None and self.has_option(section, k)}})


BOOLEAN_STATES = {"1": True, "yes": True, "true": True, "on": True, "oui": True, "0": False, "no": False, "false": False, "off": False, "non": False}


def optional_int(value):
    """Processor for a parameter representing an integer (empty is None)"""
    return int(value) if value else None


def optional_bool(value):
    """Processor for a parameter representing a boolean (empty is None)"""
    if not value:
        return None
    if value.lower() not in BOOLEAN_STATES:
        raise ValueError(f"Not a boolean: {value}")
    return BOOLEAN_STATES[value.lower()]


def load_config():
    """Load configuration from:
    - the default .ini config file
    - user provided .ini config files (using --config_file)
    - environment variables
    - a previous mlflow run configuration (using --config_mlflow)
    - parameters from the arguments
    in loading order which is the reverse order of precedence:
    each configuration overrides the previous ones.

    Exits if the --export_config flag is in the arguments.
    """

    args = argparser.parse_args()

    # Verbose mode means "debug" level of logging
    if args.verbose:
        args.loggingLevel = "DEBUG"
    # Configure logging with selected level
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S %p",
        level=args.loggingLevel,
    )

    # Load default config file first
    confparser = PostProcessedConfigParser(interpolation=ExtendedInterpolation())
    confparser.setProcessors(
        {
            "chunk_size": optional_int,
            "chunk_overlap": optional_int,
            "max_pages": optional_int,
            "force_rebuild": optional_bool,
            "markdown_split": optional_bool,
            "use_tokenizer_to_chunk": optional_bool,
            "separators": ast.literal_eval,
        }
    )
    default_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
    confparser.read(default_config_file)

    # Override DEFAULT values with custom user provided config file (if any)
    if args.config_file is not None:
        confparser.read(args.config_file)
    # Note: it is best to avoid defaults in the argument parser
    # since they will override the default values from ini file

    # Override DEFAULT values with environment variables
    # Note: in order not to leak sensitive info from environment,
    # only the variables already in the config file are overwritten
    confparser.update_dict(os.environ)

    # If a mlflow run ID is provided, override all with
    if args.mlflow_run_id:
        mlflow.tracking.MlflowClient(tracking_uri=confparser.get("DEFAULT", "mlflow_tracking_uri"))
        mlflow.set_experiment(confparser.get("DEFAULT", "experiment_name"))
        mlflow_params = mlflow.get_run(args.mlflow_run_id).data.params
        mlflow_params.pop("experiment_name", None)
        mlflow_params.pop("mlflow_tracking_uri", None)
        confparser.update_dict(mlflow_params)

    # Override DEFAULT values with user provided flags
    confparser.update_dict(vars(args))

    # If export_config is set, print out config and exit
    if args.export_config:
        confparser.write(sys.stdout)
        exit()

    return confparser
