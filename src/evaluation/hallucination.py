import re
import requests
from typing import List


# Function to extract URLs
def extract_urls(text):
    return re.findall(r"\((https?://[^\)]+)\)", text)


# Function to check URL status
def check_url(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def check_hallucination_rate(list_answers_generative: List[str]):
    # Main checking loop
    results = []
    for idx, text in enumerate(list_answers_generative):
        urls = extract_urls(text)
        for url in urls:
            valid = check_url(url)
            results.append({"element_index": idx, "url": url, "valid": valid})

    return compute_hallucination_rate(results)


def compute_hallucination_rate(results):
    """
    Computes the hallucination rate from a list of URL check results.

    Args:
        results (list of dict): Each dict should have a 'valid' key (True/False).

    Returns:
        float: Hallucination rate in percentage.
    """
    if not results:
        return 0.0  # Avoid division by zero if there are no links

    total_links = len(results)
    broken_links = sum(1 for r in results if not r["valid"])

    hallucination_rate = (broken_links / total_links) * 100
    return hallucination_rate
