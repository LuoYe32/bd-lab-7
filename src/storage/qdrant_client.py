from qdrant_client import QdrantClient

from src.config.config import QdrantConfig


class QdrantClientFactory:
    def __init__(self, config: QdrantConfig):
        self.config = config

    def create_client(self) -> QdrantClient:
        return QdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
        )