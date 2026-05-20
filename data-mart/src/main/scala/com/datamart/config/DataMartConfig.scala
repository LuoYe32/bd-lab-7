package com.datamart.config

import com.typesafe.config.{Config, ConfigFactory}

import scala.collection.JavaConverters._

case class QdrantSettings(
  url: String,
  apiKey: String,
  inputCollection: String,
  clusteredCollection: String,
  scrollBatchSize: Int,
  uploadBatchSize: Int,
  vectorSize: Int
)

case class DataSettings(
  selectedColumns: List[String],
  rowLimit: Int,
  csvPath: String,
  csvSeparator: String,
  preprocessedPath: String,
  resultsPath: String
)

case class SparkSettings(
  appName: String,
  master: String,
  driverMemory: String
)

case class DataMartConfig(
  qdrant: QdrantSettings,
  data: DataSettings,
  spark: SparkSettings
)

object DataMartConfig {
  def load(): DataMartConfig = {
    val config: Config = ConfigFactory.load()

    val qdrant = QdrantSettings(
      url                 = config.getString("qdrant.url"),
      apiKey              = config.getString("qdrant.api-key"),
      inputCollection     = config.getString("qdrant.input-collection"),
      clusteredCollection = config.getString("qdrant.clustered-collection"),
      scrollBatchSize     = config.getInt("qdrant.scroll-batch-size"),
      uploadBatchSize     = config.getInt("qdrant.upload-batch-size"),
      vectorSize          = config.getInt("qdrant.vector-size")
    )

    val data = DataSettings(
      selectedColumns  = config.getStringList("data.selected-columns").asScala.toList,
      rowLimit         = config.getInt("data.row-limit"),
      csvPath          = config.getString("data.csv-path"),
      csvSeparator     = config.getString("data.csv-separator"),
      preprocessedPath = config.getString("data.preprocessed-path"),
      resultsPath      = config.getString("data.results-path")
    )

    val spark = SparkSettings(
      appName      = config.getString("spark.app-name"),
      master       = config.getString("spark.master"),
      driverMemory = config.getString("spark.driver-memory")
    )

    DataMartConfig(qdrant, data, spark)
  }
}
