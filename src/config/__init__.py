import argparse
import ast
import configparser
import logging
import os
import sys

# https://stackoverflow.com/questions/3609852/which-is-the-best-way-to-allow-configuration-options-be-overridden-at-the-comman
# https://stackoverflow.com/questions/48538581/argparse-defaults-from-file
# https://gist.github.com/drmalex07/9995807

# https://docs.python.org/3/library/configparser.html
# https://docs.python.org/3/library/argparse.html


def str_to_list(arg):
    # Convert the argument string to a list
    return ast.literal_eval(arg)


# PARSER FOR USER LEVEL ARGUMENTS --------------------------------

argparser = argparse.ArgumentParser(description="Chroma building parameters")
argparser.add_argument(
    "--experiment_name",
    type=str,
    default="default",
    help="""
    Name of the experiment.
    """,
)
argparser.add_argument(
    "--data_raw_s3_path",
    type=str,
    default="data/raw_data/applishare_solr_joined.parquet",
    help="""
    Path to the raw data.
    Default to data/raw_data/applishare_solr_joined.parquet
    """,
)
argparser.add_argument(
    "--collection_name",
    type=str,
    default="insee_data",
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
    type=str_to_list,
    default=r"['\n\n', '\n', '.', ' ', '']",
    help="List separators to split the text",
)
argparser.add_argument(
    "--embedding_model",
    type=str,
    dest="emb_model",
    default="OrdalieTech/Solon-embeddings-large-0.1",
    help="""
    Embedding model.
    Should be a huggingface model.
    Defaults to OrdalieTech/Solon-embeddings-large-0.1
    """,
)
argparser.add_argument(
    "--max_pages",
    type=int,
    default=None,
    help="""
    Maximum number of pages to use for the vector database.
    """,
)
argparser.add_argument(
    "--chunk_size",
    type=str,
    default=None,
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
    default="cuda",
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
    metavar="FILE",
    help="""
    Specify a config file from which parameters can be read
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

# Load default config file first
confparser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
default_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
confparser.read(default_config_file)

# To load defaults from the default ini file rather than use the hardcoded defaults above
argparser.set_defaults(**confparser["DEFAULT"])


def load_config():
    args = argparser.parse_args()

    # Verbose mode means "debug" level of logging
    if args.pop("verbose"):
        args.loggingLevel = "DEBUG"
    # Configure logging with selected level
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S %p",
        level=args.loggingLevel,
    )

    # If args is set
    if args.config_export:
        confparser.write(sys.stdout)
        exit()

    # Override (and complete) DEFAULT values with (all...) environment variables
    confparser.read_dict({"DEFAULT": os.environ})
    # Override DEFAULT values with custom user provided config file (if any)
    if args.config_file is not None:
        confparser.read(args.config_file)
    # Override DEFAULT values with user provided flags
    confparser.read_dict(vars(args))
    return confparser
