import inspect
import os
from dataclasses import dataclass

import mlflow
import toml
from confz import CLArgSource, ConfigSource, DataSource, EnvSource, FileSource
from confz.base_config import BaseConfigMetaclass
from confz.loaders import Loader, register_loader

from .models import FullConfig

default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.toml")


@dataclass
class MLFlowSource(ConfigSource):
    pass


class MLFlowLoader(Loader):
    @classmethod
    def populate_config(cls, config: dict, config_source: MLFlowSource):
        if config.get("mlflow_run_id") and config.get("mlflow_tracking_uri"):
            client = mlflow.tracking.MlflowClient(tracking_uri=config["mlflow_tracking_uri"])
            mlflow_params = client.get_run(config["mlflow_run_id"]).data.params
            # Do not override ML Flow loading parameters
            mlflow_params.pop("experiment_name", None)
            mlflow_params.pop("mlflow_tracking_uri", None)
            mlflow_params.pop("mlflow_run_id", None)
            cls.update_dict_recursively(config, mlflow_params)


register_loader(MLFlowSource, MLFlowLoader)


@dataclass
class TemplatePassSource(ConfigSource):
    pass


class TemplatePassLoader(Loader):
    @classmethod
    def populate_config(cls, config: dict, config_source: TemplatePassSource):
        templated_params = config.get("__templated_params__")
        if templated_params:
            for p in templated_params:
                if config.get(p):
                    cls.update_dict_recursively(config, {p: config[p].format(**config)})


register_loader(TemplatePassSource, TemplatePassLoader)


class DefaultFullConfig(FullConfig, metaclass=BaseConfigMetaclass):
    """
    Configuration class for the FullConfig model with preconfigured sources.

    Singleton mechanism:
    - DefaultFullConfig cannot be instantiated with custom keyword arguments
    - Calls to the constructor DefaultFullConfig() are cached and basically "free":
      the config is not reloaded from sources
    """

    CONFIG_SOURCES = [
        # Set default parameters from default config file
        FileSource(file=default_config_path),
        # Set parameters from config file from env
        FileSource(file_from_env="RAG_CONFIG_FILE", optional=True),
        # Set parameter xxxx directly with the XXXX env variable
        EnvSource(allow=["AWS_S3_ENDPOINT", "WORK_DIR"]),
        # Set parameter xxxx using the RAG_XXXX (case insensitive) env variable
        EnvSource(allow_all=True, prefix="RAG_"),
        # Set parameters from config file from command line argument
        FileSource(file_from_cl="--config_file", optional=True),
        # Set parameters from command line argument
        CLArgSource(
            remap={
                # Add explicit command line arguments remapping if needed
                "config_mlflow": "mlflow_run_id"
            }
        ),
        # Set parameters from a previous MLFlow run identified with its mlflow_run_id
        MLFlowSource(),
        # Final pass to template all parameters listed in __templated_params__
        TemplatePassSource(),
    ]


def custom_config(defaults: dict | None = None, overrides: dict | None = None):
    """
    Load a configuration from files, environment and command line argument but:
    - Default values are taken from [defaults] (if specified) rather than from the default file
    - All values are overriden with [overrides] (if specified)
    """
    defaults = {k.lower(): v for k, v in defaults.items()} if defaults else {}
    overrides = {k.lower(): v for k, v in overrides.items()} if overrides else {}
    return FullConfig(
        config_sources=[
            FileSource(file=default_config_path),  # Load defaults
            DataSource(data=defaults),  # Override default with custom defaults
        ]
        + DefaultFullConfig.CONFIG_SOURCES[1:-1]  # Load all other sources
        + [
            DataSource(data=overrides),  # Override with custom overrides
            TemplatePassSource(),  # Final templating pass
        ],
        experiment_name="test",
    )


class Configurable:
    """
    Decorator for function with a special "configuration" argument.

    """

    # The decorator is initialised with the configuration argument name
    def __init__(self, config_param: str = "config"):
        self.config_param = config_param

    def __call__(self, f):
        # The original arguments from the annotated function
        declared_parameters = inspect.signature(f).parameters
        # The configuration argument's annotation (required) and default value (optional)
        config_default = declared_parameters[self.config_param].default
        config_class = declared_parameters[self.config_param].annotation
        # All fields from the config's class
        config_parameters = config_class.model_fields.keys()
        # All config fields without the ones already explicitely specified in the decorated function's arguments
        overridable_params = set(config_parameters) - set(declared_parameters.keys())

        # The returned function
        def new_f(*args, **kwargs):
            # Get original config argument from kwargs (and remove it) or use default
            orig_config = kwargs.pop(self.config_param, config_default)
            # TODO: do not hcange anything if no special overriding argument were provided!

            # Dict with the original config overriden with keyword args (which are removed from the kwargs dict)
            new_config_params = {
                k: kwargs.pop(k) if k in kwargs and k in overridable_params else getattr(orig_config, k)
                for k in config_parameters
            }
            # Updated config object buily by passing overriden parameters to the config_class constructor
            # NOTE: this only works if the class annotation of config can be built this way (e.g. pydantic BaseModel)
            new_config = config_class(**new_config_params)
            # Set the config_param to the updated object
            kwargs[self.config_param] = new_config
            # Call the original function with the updated config object
            return f(*args, **kwargs)

        return new_f


# Example:
# from src.config.config import config_test
# config_test() # Return the default 'default' unless env variable (or others) set it otherwise
# config_test(experiment_name="test") # Returns 'test'
# config_test(experiment_name=3) # Fails, as it should, since the config_class cannot build using ill-typed arguments
# config_test(toto=3)            # Fails, as it should, with "unexpected keyword argument"
@Configurable("config_arg")
def config_test(config_arg: FullConfig = DefaultFullConfig()) -> str:  # noqa: B008
    return config_arg.experiment_name


if __name__ == "__main__":
    print(toml.dumps(vars(FullConfig())))
