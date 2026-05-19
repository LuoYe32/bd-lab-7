from pyspark.sql import SparkSession

from src.config.config import ProjectConfig


class SparkManager:
    def __init__(self):
        self.config = ProjectConfig()
        self.spark: SparkSession | None = None

    def create_session(self) -> SparkSession:
        if self.spark is None:
            spark_config = self.config.spark

            self.spark = (
                SparkSession.builder
                .appName(spark_config.app_name)
                .master(spark_config.master)

                .config("spark.driver.memory", spark_config.driver_memory)
                .config("spark.executor.memory", spark_config.executor_memory)

                .config("spark.sql.shuffle.partitions", str(spark_config.shuffle_partitions))
                .config("spark.serializer", spark_config.serializer)
                .config("spark.sql.adaptive.enabled", str(spark_config.adaptive_enabled).lower())

                .config("spark.sql.autoBroadcastJoinThreshold", "10MB")
                .config("spark.sql.inMemoryColumnarStorage.compressed", "true")

                .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
                .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")

                .getOrCreate()
            )

        return self.spark

    def stop_session(self) -> None:
        if self.spark is not None:
            self.spark.stop()
            self.spark = None