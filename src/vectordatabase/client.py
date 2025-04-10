from .chroma import create_client_and_collection_chroma, get_number_docs_collection_chroma
from .qdrant import create_client_and_collection_qdrant, get_number_docs_collection_qdrant


def create_client_and_collection(url: str, collection_name: str, engine="qdrant", **kwargs):
    if engine not in ["qdrant", "chroma"]:
        raise ValueError("Only Qdrant and Chroma database are supported")

    args_to_pass = {
        "url": url,
        "collection_name": collection_name,
    }

    if engine == "qdrant":
        client = create_client_and_collection_qdrant(
            api_key=kwargs["api_key"],
            model_max_len=kwargs["model_max_len"],
            vector_name=kwargs["vector_name"],
            **args_to_pass,
        )
    else:
        client = create_client_and_collection_chroma(**args_to_pass)

    return client


def get_number_docs_collection(client, collection_name: str, engine="qdrant"):
    collection_func = get_number_docs_collection_qdrant
    if engine == "chroma":
        collection_func = get_number_docs_collection_chroma
    return collection_func(client=client, collection_name=collection_name)
