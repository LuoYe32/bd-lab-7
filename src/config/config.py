from pydantic import Field
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class SparkConfig(BaseSettings):
    app_name: str = "OpenFoodFactsClustering"
    master: str = "local[*]"

    driver_memory: str = "4g"
    executor_memory: str = "4g"

    shuffle_partitions: int = 8
    adaptive_enabled: bool = True
    serializer: str = "org.apache.spark.serializer.KryoSerializer"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class DataConfig(BaseSettings):
    data_path: str = "data/raw/products.csv"
    model_path: str = "data/processed/kmeans_model"
    preprocessed_path: str = "data/processed/preprocessed.parquet"
    predictions_path: str = "data/processed/predictions.parquet"

    selected_columns: List[str] = Field(default_factory=lambda: [
        "energy_100g",
        "fat_100g",
        "carbohydrates_100g",
        "sugars_100g",
        "proteins_100g",
    ])

    row_limit: int = 1000
    csv_separator: str = "\t"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ModelConfig(BaseSettings):
    k_clusters: int = 5
    seed: int = 51

    features_col: str = "features"
    scaled_features_col: str = "scaled_features"
    prediction_col: str = "prediction"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class QdrantConfig(BaseSettings):
    url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    api_key: str = Field(default="", alias="QDRANT_API_KEY")

    input_collection: str = Field(default="products_raw", alias="QDRANT_INPUT_COLLECTION")
    preprocessed_collection: str = Field(default="products_preprocessed", alias="QDRANT_PREPROCESSED_COLLECTION")
    output_collection: str = Field(default="products_clustered", alias="QDRANT_OUTPUT_COLLECTION")

    vector_size: int = 5
    upload_batch_size: int = 512
    scroll_batch_size: int = 1000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class DataMartConfig(BaseSettings):
    jar_path: str = Field(
        default="data-mart/target/scala-2.12/data-mart-assembly.jar",
        alias="DATA_MART_JAR",
    )
    spark_home: str = Field(default="", alias="SPARK_HOME")

    @property
    def resolved_spark_home(self) -> str:
        if self.spark_home:
            return self.spark_home
        try:
            import pyspark
            import os
            return os.path.dirname(pyspark.__file__)
        except ImportError:
            raise RuntimeError(
                "SPARK_HOME is not set, and pyspark was not found. "
                "You should specify SPARK_HOME in .env or install pyspark."
            )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class ProjectConfig(BaseSettings):
    spark: SparkConfig = Field(default_factory=SparkConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    data_mart: DataMartConfig = Field(default_factory=DataMartConfig)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")