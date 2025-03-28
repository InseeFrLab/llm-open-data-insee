from qdrant_client import QdrantClient


def get_number_docs_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
):
    collection_info = qdrant_client.get_collection(collection_name=collection_name)
    return collection_info.points_count
