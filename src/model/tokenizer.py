import requests


def tokenize_prompt(prompt: str, model: str, api_root_url: str):
    api_root_url = api_root_url.replace("v1/", "")
    url = f"{api_root_url}/tokenize/"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "add_special_tokens": True, "additionalProp1": {}}

    response = requests.post(url, headers=headers, json=payload)

    if response.ok:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None


# # TEST EMBEDING SIZE
# import os
# documents_chunk = documents

# size_chunked = {
#     index: tokenize_prompt(
#     content.page_content, model= embedding_model, api_root_url=os.getenv("URL_GENERATIVE_MODEL")
# ).get('count') for index, content in enumerate(documents_chunk)}


# documents_chunk[1].page_content

# documents[0].page_content
