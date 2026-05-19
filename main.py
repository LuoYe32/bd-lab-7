from src.config.config import ProjectConfig
from src.data.qdrant_loader import QdrantDataLoader
from src.pipeline.clustering_pipeline import ClusteringPipeline
from src.pipeline.data_upload_pipeline import DataUploadPipeline
from src.storage.qdrant_client import QdrantClientFactory
from src.utils.logger import LoggerFactory
from src.utils.spark_manager import SparkManager


def main() -> None:
    logger = LoggerFactory.get_logger("Main")

    config = ProjectConfig()

    spark_manager = SparkManager()
    spark = spark_manager.create_session()

    try:
        client = QdrantClientFactory(config.qdrant).create_client()

        loader = QdrantDataLoader(
            spark=spark,
            client=client,
            collection_name=config.qdrant.input_collection,
            selected_columns=config.data.selected_columns,
        )

        if not loader.has_data():
            logger.info(
                "Qdrant collection '%s' is empty. Uploading source data...",
                config.qdrant.input_collection,
            )

            upload_pipeline = DataUploadPipeline()
            upload_pipeline.run()

        else:
            logger.info(
                "Qdrant collection '%s' already contains data.",
                config.qdrant.input_collection,
            )

        clustering_pipeline = ClusteringPipeline()
        clustering_pipeline.run()

    finally:
        spark_manager.stop_session()


if __name__ == "__main__":
    main()