"""
Utilitary functions.
"""

from typing import Dict
import s3fs
import os


fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://" + os.environ["AWS_S3_ENDPOINT"]}
)


def create_ls_task(
    question: str,
    answer: str,
) -> Dict[str, str]:
    """
    Create Label Studio task .json for a question/answer pair.

    Args:
        question (str): Question.
        answer (str): Answer.

    Returns:
        Dict[str, str]: Label Studio json task.
    """
    return {
        "data": {
            "question": question,
            "answer": answer,
        }
    }
