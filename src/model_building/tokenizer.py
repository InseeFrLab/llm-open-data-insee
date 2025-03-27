import requests


def tokenize_prompt(prompt: str, model: str, api_root_url: str):
    url = f"{api_root_url}/tokenize/"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "add_special_tokens": True, "additionalProp1": {}}

    response = requests.post(url, headers=headers, json=payload)

    if response.ok:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None
