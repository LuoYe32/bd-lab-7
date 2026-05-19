from qdrant_client import QdrantClient
from pyspark.sql import DataFrame, SparkSession


class QdrantDataLoader:
    def __init__(
        self,
        spark: SparkSession,
        client: QdrantClient,
        collection_name: str,
        selected_columns: list[str],
        batch_size: int = 1000,
    ):
        self.spark = spark
        self.client = client
        self.collection_name = collection_name
        self.selected_columns = selected_columns
        self.batch_size = batch_size

    def has_data(self) -> bool:
        if not self.client.collection_exists(self.collection_name):
            return False

        collection_info = self.client.get_collection(self.collection_name)

        return collection_info.points_count > 0

    def load(self) -> DataFrame:
        rows: list[dict] = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=self.batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}

                row = {
                    column_name: payload.get(column_name)
                    for column_name in self.selected_columns
                }

                rows.append(row)

            if offset is None:
                break

        if not rows:
            raise ValueError(
                f"No data found in Qdrant collection: {self.collection_name}"
            )

        return self.spark.createDataFrame(rows)