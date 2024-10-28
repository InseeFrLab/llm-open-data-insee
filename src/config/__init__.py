import argparse
import ast
import configparser
import logging
import os
import sys


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


class PostProcessedConfigParser(configparser.ConfigParser):
    """
    ConfigParser with registered internal post-processors that are run on templated options.
    Only the `get` method is modified. Sections of this ConfigParser should work fine as they
    rely on this method to access options.
    """

    def __init__(self, *args, **kwargs):
        super(PostProcessedConfigParser, self).__init__(*args, **kwargs)
        self._processors = {}

    def setProcessor(self, option, proc):
        self._processors[self.optionxform(option)] = proc

    def setProcessors(self, procs):
        for opt, proc in procs.items():
            self.setProcessor(opt, proc)

    def get(self, section, option, **kwargs):
        tpl_opt = super(PostProcessedConfigParser, self).get(section, option, **kwargs)
        proc = self._processors.get(self.optionxform(option))
        return proc(tpl_opt) if proc and tpl_opt else tpl_opt


BOOLEAN_STATES = {'1': True,  'yes': True,  'true': True,   'on': True,  'oui': True,
                  '0': False, 'no': False, 'false': False, 'off': False, 'non': False}


def optional_int(value):
    """ Processor for a parameter representing an integer (empty is None) """
    return int(value) if value else None


def optional_bool(value):
    """ Processor for a parameter representing a boolean (empty is None) """
    if value is None:
        return None
    if value.lower() not in BOOLEAN_STATES:
        raise ValueError('Not a boolean: %s' % value)
    return BOOLEAN_STATES[value.lower()]


# Load default config file first
confparser = PostProcessedConfigParser(interpolation=configparser.ExtendedInterpolation())
confparser.setProcessors(
    {
        "chunk_size": optional_int,
        "chunk_overlap": optional_int,
        "max_pages": optional_int,
        "force_rebuild": optional_bool,
        "markdown_split": optional_bool,
        "use_tokenizer_to_chunk": optional_bool,
        "separators": ast.literal_eval
    }
)
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

    # Override DEFAULT values with environment variables
    # Note: in order not to leak sensitive info from environment,
    # only the variables already in the config file are overwritten
    env_remap = {k: os.environ[k] for k in os.environ if confparser["DEFAULT"].has_option(k)}
    confparser.read_dict({"DEFAULT": env_remap})

    # Override DEFAULT values with custom user provided config file (if any)
    if args.config_file is not None:
        confparser.read(args.config_file)
    # Override DEFAULT values with user provided flags
    confparser.read_dict(vars(args))
    return confparser
