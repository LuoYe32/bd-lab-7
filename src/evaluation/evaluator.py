from pyspark.sql import DataFrame
from pyspark.ml.evaluation import ClusteringEvaluator


class ClusteringModelEvaluator:
    def __init__(
        self,
        features_col: str,
        prediction_col: str,
        metric_name: str = "silhouette",
    ):
        self.features_col = features_col
        self.prediction_col = prediction_col
        self.metric_name = metric_name

    def evaluate(self, df: DataFrame) -> float:
        evaluator = ClusteringEvaluator(
            featuresCol=self.features_col,
            predictionCol=self.prediction_col,
            metricName=self.metric_name,
        )

        return evaluator.evaluate(df)