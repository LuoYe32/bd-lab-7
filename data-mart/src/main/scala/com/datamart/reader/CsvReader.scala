package com.datamart.reader

import com.datamart.config.DataMartConfig
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType


class CsvReader(config: DataMartConfig)(implicit spark: SparkSession) {

  def read(): DataFrame = {
    spark.read
      .option("header", "true")
      .option("sep", config.data.csvSeparator)
      .option("inferSchema", "true")
      .csv(config.data.csvPath)
      .select(config.data.selectedColumns.map(col): _*)
      .na.drop()
      .select(config.data.selectedColumns.map(c => col(c).cast(DoubleType).alias(c)): _*)
      .na.drop()
  }
}
