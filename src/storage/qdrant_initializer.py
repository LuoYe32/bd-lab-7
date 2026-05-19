from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


class QdrantInitializer:
    def __init__(self, client: QdrantClient):
        self.client = client

    def recreate_collection(self, collection_name: str, vector_size: int) -> None:
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )