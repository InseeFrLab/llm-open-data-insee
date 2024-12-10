import argparse
import logging
import sys
from argparse import ArgumentParser, BooleanOptionalAction

import toml
from confz import validate_all_configs

from .config import DefaultFullConfig

# PARSER FOR USER LEVEL ARGUMENTS --------------------------------


def minimal_argparser():
    argparser = ArgumentParser()
    argparser.add_argument(
        "--config_file",
        type=str,
        metavar="FILE",
        help="""
        Specify a TOML, YAML or JSON config file from which parameters can be read
        """,
    )
    argparser.add_argument(
        "--config_mlflow",
        type=str,
        metavar="RUNID",
        help="""
        Load configuration from a previous mlflow run.
        This configuration overrides config files and environment variables.
        Other explicit flags will override this configuration.
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
    return argparser


def models_only_argparser():
    argparser = minimal_argparser()
    argparser.add_argument(
        "--emb_model",
        type=str,
        help="""
        Embedding model for information retrieval.
        Should be a huggingface model.
        Defaults to OrdalieTech/Solon-embeddings-large-0.1
        """,
    )
    argparser.add_argument(
        "--llm_model",
        type=str,
        help="""
        LLM used to generate chat.
        Should be a huggingface model.
        Defaults to mistralai/Mistral-7B-Instruct-v0.2
        """,
    )
    return argparser


def simple_argparser():
    argparser = minimal_argparser()
    argparser.add_argument(
        "--work_dir",
        type=str,
        help="""
        Work directory in which the "data" folder will be created.
        """,
    )
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
        action=BooleanOptionalAction,
        help="""
        Should we use a markdown split ?
        --markdown_split yields True and --no-markdown_split yields False
        """,
    )
    argparser.add_argument(
        "--use_tokenizer_to_chunk",
        default=True,
        action=BooleanOptionalAction,
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
        help="""
        Chunk overlap
        """,
    )
    argparser.add_argument(
        "--emb_device",
        type=str,
        help="""
        Embedding device (cpu, cuda, etc.)
        """,
    )
    argparser.add_argument(
        "--force_rebuild",
        default=True,
        action=BooleanOptionalAction,
        help="""
        Should we reuse previously constructed database (--no-force_rebuild, default)  or rebuild (--force_rebuild)?
        """,
    )
    argparser.add_argument(
        "--batch_size_embedding",
        default=int,
        help="""
        Batch size for embedding in the vector database.
        """,
    )
    return argparser


def llm_argparser():
    """LLM specific argument parser"""
    argparser = simple_argparser()
    argparser.add_argument(
        "--llm_model",
        type=str,
        help="""
        LLM used to generate chat.
        Should be a huggingface model.
        Defaults to mistralai/Mistral-7B-Instruct-v0.2
        """,
    )
    argparser.add_argument(
        "--quantization",
        default=True,
        action=BooleanOptionalAction,
        help="""
        Should we use a quantized version of "model" argument ?
        --quantization yields True and --no-quantization yields False
        """,
    )
    argparser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2000,
        help="""
        The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        See https://huggingface.co/docs/transformers/main_classes/text_generation
        """,
    )
    argparser.add_argument(
        "--model_temperature",
        type=int,
        default=0.2,
        help="""
        The value used to modulate the next token probabilities.
        See https://huggingface.co/docs/transformers/main_classes/text_generation
        """,
    )
    argparser.add_argument(
        "--return_full_text",
        action=BooleanOptionalAction,
        default=True,
        help="""
        Should we return the full text ?
        --return_full_text yields True and --no-return_full_text yields False
        Default to True
        """,
    )
    argparser.add_argument(
        "--do_sample",
        action=BooleanOptionalAction,
        default=True,
        help="""
        if set to True , this parameter enables decoding strategies such as multinomial
        sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling.
        All these strategies select the next token from the probability distribution
        over the entire vocabulary with various strategy-specific adjustments.
        --do_sample yields True and --no-do_sample yields False
        Default to True
        """,
    )
    argparser.add_argument(
        "--reranking_method",
        type=str,
        default=None,
        help="""
        Reranking document relevancy after retrieval phase.
        Defaults to None (no reranking)
        """,
    )
    argparser.add_argument(
        "--topk_stats",
        type=int,
        default=5,
        help="""
        Number of links considered to evaluate retriever quality.
        """,
    )
    return argparser


def process_args(argparser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    args = (argparser if argparser is not None else minimal_argparser()).parse_args()

    # Configure logging with selected level
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S %p",
        level="DEBUG" if args.verbose else args.loggingLevel,
    )

    # Immediately load all configuration from config files, environment, command line arguments and/or mlflow run id
    validate_all_configs()
    if args.export_config:
        # If export_config is set, simply print out the loaded config and exit
        toml.dump(vars(DefaultFullConfig()), sys.stdout)
        exit()
    return args
