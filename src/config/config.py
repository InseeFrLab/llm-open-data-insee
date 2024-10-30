import argparse
import ast
import logging
import os
import sys
from configparser import ConfigParser, ExtendedInterpolation

import mlflow


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
        options = {k: str(v) for k, v in d.items() if v is not None and self.has_option(section, k)}
        self.read_dict({section: options})


BOOLEAN_STATES = {"1": True, "yes": True, "true": True, "on": True, "oui": True, "0": False, "no": False, "false": False, "off": False, "non": False}


def optional_int(value: str) -> int | None:
    """Processor for a parameter representing an integer (empty is None)"""
    return int(value) if value else None


def optional_bool(value: str) -> bool | None:
    """Processor for a parameter representing a boolean (empty is None)"""
    if not value:
        return None
    if value.lower() not in BOOLEAN_STATES:
        raise ValueError(f"Not a boolean: {value}")
    return BOOLEAN_STATES[value.lower()]


# Global variables to access default config: confparser and default_config proxy
confparser = PostProcessedConfigParser(interpolation=ExtendedInterpolation())
default_config = confparser[confparser.default_section]


def load_config(argparser: argparse.ArgumentParser | None = None) -> PostProcessedConfigParser:
    """Load configuration from:
    - the default .ini config file
    - environment variables
    - a previous mlflow run configuration (using --config_mlflow)
    - user-provided .ini config files (using --config_file)
    - parameters from the arguments
    in loading order which is the reverse order of precedence:
    each configuration overrides the previous ones.

    Prints configuration and exits if the --export_config flag is in the arguments.
    Prints help and exits if the --help flag is in the arguments.
    """
    # Custom PostProcessors
    confparser.setProcessors(
        {
            "chunk_size": optional_int,
            "chunk_overlap": optional_int,
            "max_pages": optional_int,
            "force_rebuild": optional_bool,
            "markdown_split": optional_bool,
            "use_tokenizer_to_chunk": optional_bool,
            "separators": ast.literal_eval,
            "max_new_tokens": optional_int,
            "quantization": optional_int,
            "mlflow_load_artifacts": optional_bool,
        }
    )
    # Load default config file first
    default_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
    confparser.read(default_config_file)

    # Override values from DEFAULT section with environment variables
    # Note: in order not to leak sensitive info from environment,
    # only the variables already in the section (no credentials) are overwritten
    confparser.update_dict(os.environ)

    if argparser is not None:
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

        # Override DEFAULT values with custom user provided config file (if any)
        if args.config_file is not None:
            confparser.read(args.config_file)
        # Note: it is best to avoid defaults in the argument parser
        # since they will override the default values from ini file

        # If a mlflow run ID is provided, load parameters
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
