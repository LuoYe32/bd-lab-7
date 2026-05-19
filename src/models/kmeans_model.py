from pyspark.sql import DataFrame
from pyspark.ml.clustering import KMeans, KMeansModel


class KMeansClusteringModel:
    def __init__(
        self,
        k: int,
        seed: int,
        features_col: str,
        prediction_col: str,
    ):
        self.k = k
        self.seed = seed
        self.features_col = features_col
        self.prediction_col = prediction_col
        self.model: KMeansModel | None = None

    def train(self, df: DataFrame) -> KMeansModel:
        estimator = KMeans(
            k=self.k,
            seed=self.seed,
            featuresCol=self.features_col,
            predictionCol=self.prediction_col,
        )

        self.model = estimator.fit(df)
        return self.model

    def predict(self, df: DataFrame) -> DataFrame:
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        return self.model.transform(df)

    def get_cluster_centers(self):
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        return self.model.clusterCenters()

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        self.model.write().overwrite().save(path)