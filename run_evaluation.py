import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import s3fs
from langchain_core.prompts import PromptTemplate

from src.chain_building import build_chain_validator
from src.chain_building.build_chain import build_chain
from src.config import RAGConfig, llm_argparser, load_config
from src.db_building import chroma_topk_to_df, load_retriever, load_vector_database
from src.evaluation import (
    answer_faq_by_bot,
    compare_performance_reranking,
    evaluate_question_validator,
    transform_answers_bot,
)
from src.model_building import build_llm_model
from src.utils.formatting_utilities import get_chatbot_template

# Logging configuration
logger = logging.getLogger(__name__)


def run_evaluation(filesystem: s3fs.S3FileSystem, config: Mapping[str, Any] = vars(RAGConfig())) -> None:
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run():
        # Logging the full configuration to mlflow
        mlflow.log_params(dict(config))

        # INPUT: FAQ THAT WILL BE USED FOR EVALUATION -----------------
        faq = pd.read_parquet(config["faq_s3_uri"], filesystem=filesystem)
        # Extract all URLs from the 'sources' column
        faq["urls"] = faq["sources"].str.findall(r"https?://www\.insee\.fr[^\s]*").apply(lambda s: ", ".join(s))

        # ------------------------
        # I - LOAD VECTOR DATABASE

        # Ensure correct database is used
        db = load_vector_database(filesystem, config=config)

        # ------------------------
        # II - CREATING RETRIEVER

        logger.info(f"Training retriever {80*'='}")

        mlflow.log_text(config["RAG_PROMPT_TEMPLATE"], "rag_prompt.md")

        # Load LLM in session
        llm, tokenizer = build_llm_model(
            model_name=config["llm_model"],
            load_LLM_config=True,
            streaming=False,
            config=config,
        )

        logger.info("Logging an example of tokenized text")
        query = "Quels sont les chiffres du chômages en 2023 ?"
        mlflow.log_text(
            f"{query} \n ---------> \n {', '.join(tokenizer.tokenize(query))}",
            "example_tokenizer.json",
        )

        retriever, vectorstore = load_retriever(
            vectorstore=db,
            retriever_params={"search_type": "similarity", "search_kwargs": {"k": 30}},
            config=config,
        )

        # Log retriever
        retrieved_docs = retriever.invoke("Quels sont les chiffres du chômage en 2023 ?")
        result_retriever_raw = chroma_topk_to_df(retrieved_docs)
        mlflow.log_table(
            data=result_retriever_raw,
            artifact_file="retrieved_documents_retriever_raw.json",
        )

        # ------------------------
        # III - QUESTION VALIDATOR

        logger.info("Testing the questions that are accepted/refused by our agent")

        validator = build_chain_validator(evaluator_llm=llm, tokenizer=tokenizer)
        validator_answers = evaluate_question_validator(validator=validator)
        true_positive_validator = validator_answers.loc[validator_answers["real"], "real"].mean()
        true_negative_validator = 1 - (validator_answers.loc[~validator_answers["real"], "real"].mean())
        mlflow.log_metric("validator_true_positive", 100 * true_positive_validator)
        mlflow.log_metric("validator_negative", 100 * true_negative_validator)

        # ------------------------
        # IV - RERANKER

        reranking_method = config.get("reranking_method")

        if reranking_method is not None:
            logger.info(f"Applying reranking {80*'='}")
            logger.info(f"Selected method: {reranking_method}")

            # Define a langchain prompt template
            RAG_PROMPT_TEMPLATE_RERANKER = tokenizer.apply_chat_template(
                get_chatbot_template(), tokenize=False, add_generation_prompt=True
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE_RERANKER)

            mlflow.log_dict(get_chatbot_template(config)[0], "chatbot_template.json")

            chain = build_chain(
                retriever=retriever,
                prompt=prompt,
                llm=llm,
                reranker=reranking_method,
            )
        else:
            logger.info(f"Skipping reranking since value is None {80*'='}")

        # ------------------------
        # V - EVALUATION

        logger.info(f"Evaluating model performance against expectations {80*'='}")

        if reranking_method is None:
            answers_bot = answer_faq_by_bot(retriever, faq)
            eval_reponses_bot, answers_bot_topk = transform_answers_bot(answers_bot, k=int(config.get("topk_stats")))
        else:
            answers_bot_before_reranker = answer_faq_by_bot(retriever, faq)
            eval_reponses_bot_before_reranker, answers_bot_topk_before_reranker = transform_answers_bot(
                answers_bot_before_reranker, k=5
            )
            answers_bot_after_reranker = answer_faq_by_bot(chain, faq)
            eval_reponses_bot_after_reranker, answers_bot_topk_after_reranker = transform_answers_bot(
                answers_bot_after_reranker, k=5
            )
            eval_reponses_bot = compare_performance_reranking(
                eval_reponses_bot_after_reranker, eval_reponses_bot_before_reranker
            )
            answers_bot_topk = answers_bot_topk_after_reranker

        # Compute model performance at the end of the pipeline
        document_among_topk = answers_bot_topk["cumsum_url_expected"].max()
        document_is_top = answers_bot_topk["cumsum_url_expected"].min()
        # Also compute model performance before reranking when relevant
        if reranking_method is not None:
            document_among_topk_before_reranker = answers_bot_topk_before_reranker["cumsum_url_expected"].max()
            document_is_top_before_reranker = answers_bot_topk_before_reranker["cumsum_url_expected"].min()

        # Store FAQ
        mlflow_faq_raw = mlflow.data.from_pandas(faq, source=config["faq_s3_uri"], name="FAQ_data")
        mlflow.log_input(mlflow_faq_raw, context="faq-raw")
        mlflow.log_table(data=faq, artifact_file="faq_data.json")

        # Check if document expected is in topk answers =========================
        mlflow.log_metric("document_is_first", 100 * document_is_top)
        mlflow.log_metric("document_among_topk", 100 * document_among_topk)
        mlflow.log_metrics(
            {
                f'document_in_top_{int(row["document_position"])}': 100 * row["cumsum_url_expected"]
                for _, row in answers_bot_topk.iterrows()
            }
        )
        mlflow.log_table(data=eval_reponses_bot, artifact_file="output/eval_reponses_bot.json")

        # If we used reranking, we also store performance before reranking
        if reranking_method is not None:
            mlflow.log_metric("document_is_first_before_reranker", 100 * document_is_top_before_reranker)
            mlflow.log_metric("document_among_topk_before_reranker", 100 * document_among_topk_before_reranker)
            mlflow.log_metrics(
                {
                    f'document_in_top_{int(row["document_position"])}_before_reranker': 100 * row["cumsum_url_expected"]
                    for _, row in answers_bot_topk_before_reranker.iterrows()
                }
            )

        # Log environment necessary to reproduce the experiment
        current_dir = Path(".")
        FILES_TO_LOG = list(current_dir.glob("src/db_building/*.py")) + list(current_dir.glob("src/config/*.py"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            for file_path in FILES_TO_LOG:
                relative_path = file_path.relative_to(current_dir)
                destination_path = tmp_dir_path / relative_path.parent
                destination_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(file_path, destination_path)

            # Generate requirements.txt using pipreqs
            subprocess.run(["pipreqs", str(tmp_dir_path)], check=True)

            # Log all Python files to MLflow artifact
            mlflow.log_artifacts(tmp_dir, artifact_path="environment")


if __name__ == "__main__":
    argparser = llm_argparser()
    load_config(argparser)
    assert RAGConfig().mlflow_tracking_uri is not None, "Please set the mlflow_tracking_uri parameter."
    assert os.environ.get("HF_TOKEN"), "Please set the HF_TOKEN environment variable."
    filesystem = s3fs.S3FileSystem(endpoint_url=RAGConfig().s3_endpoint_url)
    run_evaluation(filesystem)
