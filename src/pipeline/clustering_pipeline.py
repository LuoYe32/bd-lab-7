from pyspark.sql.functions import count

from src.config.config import ProjectConfig
from src.data.qdrant_loader import QdrantDataLoader
from src.evaluation.evaluator import ClusteringModelEvaluator
from src.features.builder import FeatureBuilder
from src.models.kmeans_model import KMeansClusteringModel
from src.storage.qdrant_client import QdrantClientFactory
from src.storage.qdrant_initializer import QdrantInitializer
from src.storage.qdrant_sender import QdrantResultSender
from src.utils.logger import LoggerFactory
from src.utils.spark_manager import SparkManager


class ClusteringPipeline:
    def __init__(self):
        self.config = ProjectConfig()
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

        self.spark_manager = SparkManager()
        self.client = QdrantClientFactory(self.config.qdrant).create_client()

    def run(self) -> None:
        self.logger.info("Starting clustering pipeline...")
        try:
            self._execute()
        finally:
            self.spark_manager.stop_session()

    def _execute(self) -> None:
        spark = self.spark_manager.create_session()

        loader = QdrantDataLoader(
            spark=spark,
            client=self.client,
            collection_name=self.config.qdrant.input_collection,
            selected_columns=self.config.data.selected_columns,
            batch_size=self.config.qdrant.scroll_batch_size,
        )
        feature_builder = FeatureBuilder(
            input_columns=self.config.data.selected_columns,
            features_col=self.config.model.features_col,
            scaled_features_col=self.config.model.scaled_features_col,
        )
        model = KMeansClusteringModel(
            k=self.config.model.k_clusters,
            seed=self.config.model.seed,
            features_col=self.config.model.scaled_features_col,
            prediction_col=self.config.model.prediction_col,
        )
        evaluator = ClusteringModelEvaluator(
            features_col=self.config.model.scaled_features_col,
            prediction_col=self.config.model.prediction_col,
        )

        initializer = QdrantInitializer(self.client)
        initializer.recreate_collection(
            collection_name=self.config.qdrant.output_collection,
            vector_size=self.config.qdrant.vector_size,
        )

        self.logger.info("Loading source data from Qdrant...")
        source_df = loader.load()
        self.logger.info("Loaded rows from Qdrant: %s", source_df.count())

        self.logger.info("Building Spark ML features...")
        featured_df = feature_builder.build(source_df)

        self.logger.info("Training K-Means model...")
        model.train(featured_df)

        self.logger.info("Making predictions...")
        predictions_df = model.predict(featured_df)

        self._log_evaluation(model, evaluator, predictions_df)

        self.logger.info("Sending clustering results to Qdrant...")
        sender = QdrantResultSender(
            client=self.client,
            collection_name=self.config.qdrant.output_collection,
            selected_columns=self.config.data.selected_columns,
            features_col=self.config.model.scaled_features_col,
            prediction_col=self.config.model.prediction_col,
            batch_size=self.config.qdrant.upload_batch_size,
        )
        sender.send(predictions_df)

        self.logger.info("Saving trained model...")
        model.save(self.config.data.model_path)

        self.logger.info("Pipeline finished successfully.")

    def _log_evaluation(
        self,
        model: KMeansClusteringModel,
        evaluator: ClusteringModelEvaluator,
        predictions_df,
    ) -> None:
        silhouette_score = evaluator.evaluate(predictions_df)
        self.logger.info("Silhouette score: %.4f", silhouette_score)

        self.logger.info("Cluster centers:")
        for idx, center in enumerate(model.get_cluster_centers()):
            self.logger.info("Cluster %s: %s", idx, center)

        self.logger.info("Cluster sizes:")
        predictions_df.groupBy(self.config.model.prediction_col) \
            .agg(count("*").alias("cluster_size")) \
            .orderBy(self.config.model.prediction_col) \
            .show()