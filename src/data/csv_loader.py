from pyspark.sql import DataFrame, SparkSession


class CsvDataLoader:
    def __init__(self, spark: SparkSession, file_path: str, separator: str = "\t"):
        self.spark = spark
        self.file_path = file_path
        self.separator = separator

    def load(self) -> DataFrame:
        return (
            self.spark.read
            .option("header", True)
            .option("inferSchema", True)
            .option("sep", self.separator)
            .csv(self.file_path)
        )