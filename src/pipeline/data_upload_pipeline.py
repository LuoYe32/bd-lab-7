from qdrant_client.http.models import PointStruct

from src.config.config import ProjectConfig
from src.data.csv_loader import CsvDataLoader
from src.data.preprocessing import DataPreprocessor
from src.features.builder import FeatureBuilder
from src.storage.qdrant_client import QdrantClientFactory
from src.storage.qdrant_initializer import QdrantInitializer
from src.utils.logger import LoggerFactory
from src.utils.spark_manager import SparkManager


class DataUploadPipeline:
    def __init__(self):
        self.config = ProjectConfig()
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

        self.spark_manager = SparkManager()
        self.client = QdrantClientFactory(self.config.qdrant).create_client()

    def run(self) -> None:
        spark = None

        try:
            self.logger.info("Starting data upload pipeline...")

            spark = self.spark_manager.create_session()

            loader = CsvDataLoader(
                spark=spark,
                file_path=self.config.data.data_path,
                separator=self.config.data.csv_separator,
            )

            preprocessor = DataPreprocessor(
                selected_columns=self.config.data.selected_columns,
                row_limit=self.config.data.row_limit,
            )

            feature_builder = FeatureBuilder(
                input_columns=self.config.data.selected_columns,
                features_col=self.config.model.features_col,
                scaled_features_col=self.config.model.scaled_features_col,
            )

            initializer = QdrantInitializer(self.client)
            initializer.recreate_collection(
                collection_name=self.config.qdrant.input_collection,
                vector_size=self.config.qdrant.vector_size,
            )

            raw_df = loader.load()
            processed_df = preprocessor.preprocess(raw_df)
            featured_df = feature_builder.build(processed_df)

            rows = featured_df.select(
                *self.config.data.selected_columns,
                self.config.model.scaled_features_col,
            ).collect()

            points: list[PointStruct] = []

            for idx, row in enumerate(rows):
                vector = [
                    float(value)
                    for value in row[self.config.model.scaled_features_col].toArray().tolist()
                ]

                payload = {
                    column_name: float(row[column_name])
                    for column_name in self.config.data.selected_columns
                }

                points.append(
                    PointStruct(
                        id=idx,
                        vector=vector,
                        payload=payload,
                    )
                )

                if len(points) >= self.config.qdrant.upload_batch_size:
                    self.client.upsert(
                        collection_name=self.config.qdrant.input_collection,
                        points=points,
                    )
                    points = []

            if points:
                self.client.upsert(
                    collection_name=self.config.qdrant.input_collection,
                    points=points,
                )

            self.logger.info(
                "Uploaded %s records to Qdrant collection '%s'",
                len(rows),
                self.config.qdrant.input_collection,
            )

        except Exception as exc:
            self.logger.exception("Data upload pipeline failed: %s", exc)
            raise

        finally:
            self.spark_manager.stop_session()