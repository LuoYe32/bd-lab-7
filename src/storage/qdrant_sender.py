from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql import DataFrame


class QdrantResultSender:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        selected_columns: list[str],
        features_col: str,
        prediction_col: str,
        batch_size: int = 512,
    ):
        self.client = client
        self.collection_name = collection_name
        self.selected_columns = selected_columns
        self.features_col = features_col
        self.prediction_col = prediction_col
        self.batch_size = batch_size

    @staticmethod
    def _vector_to_list(vector: DenseVector | SparseVector) -> list[float]:
        return [float(value) for value in vector.toArray().tolist()]

    def send(self, df: DataFrame) -> None:
        rows = df.select(
            *self.selected_columns,
            self.features_col,
            self.prediction_col,
        ).collect()

        points: list[PointStruct] = []

        for idx, row in enumerate(rows):
            vector = self._vector_to_list(row[self.features_col])

            payload = {
                column_name: float(row[column_name])
                for column_name in self.selected_columns
            }
            payload["cluster"] = int(row[self.prediction_col])

            points.append(
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload=payload,
                )
            )

            if len(points) >= self.batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                points = []

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )