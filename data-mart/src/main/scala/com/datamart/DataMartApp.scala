package com.datamart

import com.datamart.config.DataMartConfig
import com.datamart.preprocessing.DataPreprocessor
import com.datamart.reader.{CsvReader, QdrantReader}
import com.datamart.writer.QdrantWriter
import org.apache.spark.sql.{SaveMode, SparkSession}


object DataMartApp {

  def main(args: Array[String]): Unit = {
    val mode   = if (args.nonEmpty) args(0) else "preprocess"
    val config = DataMartConfig.load()

    implicit val spark: SparkSession = SparkSession.builder()
      .appName(s"DataMart-$mode")
      .master(config.spark.master)
      .config("spark.driver.memory", config.spark.driverMemory)
      .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
      .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    try {
      mode match {
        case "upload"        => runUpload(config)
        case "write-results" => runWriteResults(config)
        case _               => runPreprocess(config)
      }
    } finally {
      spark.stop()
    }
  }


  private def runUpload(config: DataMartConfig)(implicit spark: SparkSession): Unit = {
    println("[DataMart:upload] Reading CSV...")
    val df = new CsvReader(config).read()
    println(s"[DataMart:upload] Loaded ${df.count()} rows from CSV")

    val writer = new QdrantWriter(config, config.qdrant.inputCollection)
    writer.recreateCollection()
    writer.write(df)

    println("[DataMart:upload] Done.")
  }

  private def runPreprocess(config: DataMartConfig)(implicit spark: SparkSession): Unit = {
    println("[DataMart:preprocess] Reading from Qdrant...")
    val rawDf = new QdrantReader(config).read()
    println(s"[DataMart:preprocess] Loaded ${rawDf.count()} rows")

    val processedDf = new DataPreprocessor(config.data).preprocess(rawDf)
    println(s"[DataMart:preprocess] After preprocessing: ${processedDf.count()} rows")

    println(s"[DataMart:preprocess] Writing to ${config.data.preprocessedPath}...")
    processedDf.write.mode(SaveMode.Overwrite).parquet(config.data.preprocessedPath)

    println("[DataMart:preprocess] Done.")
  }

  private def runWriteResults(config: DataMartConfig)(implicit spark: SparkSession): Unit = {
    println(s"[DataMart:write-results] Reading predictions from ${config.data.resultsPath}...")
    val df = spark.read.parquet(config.data.resultsPath)
    println(s"[DataMart:write-results] Loaded ${df.count()} rows")

    val writer = new QdrantWriter(config, config.qdrant.clusteredCollection)
    writer.recreateCollection()
    writer.writeWithCluster(df, predictionCol = "prediction")

    println("[DataMart:write-results] Done.")
  }
}
