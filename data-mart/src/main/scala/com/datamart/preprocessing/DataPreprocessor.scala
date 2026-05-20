package com.datamart.preprocessing

import com.datamart.config.DataSettings
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, when}


class DataPreprocessor(settings: DataSettings) {

  private val lowerQuantile = 0.01
  private val upperQuantile = 0.99
  private val relativeError  = 0.01 

  def preprocess(df: DataFrame): DataFrame = {
    var result = df
    result = selectColumns(result)
    result = removeMissingValues(result)
    result = removeNegativeValues(result)
    result = removeBasicOutliers(result)
    result = removeOutliersPercentile(result)
    result = limitRows(result)
    result = addEnergyTier(result)
    result
  }

  private def selectColumns(df: DataFrame): DataFrame =
    df.select(settings.selectedColumns.map(col): _*)

  private def removeMissingValues(df: DataFrame): DataFrame =
    df.na.drop()

  private def removeNegativeValues(df: DataFrame): DataFrame = {
    val condition = settings.selectedColumns
      .map(name => col(name) >= 0)
      .reduce(_ && _)
    df.filter(condition)
  }

  private def removeBasicOutliers(df: DataFrame): DataFrame =
    df.filter(
      col("energy_100g")       < 5000 &&
      col("fat_100g")          < 100  &&
      col("carbohydrates_100g") < 100  &&
      col("sugars_100g")       < 100  &&
      col("proteins_100g")     < 100
    )

  private def removeOutliersPercentile(df: DataFrame): DataFrame = {
    val condition = settings.selectedColumns.map { name =>
      val Array(lower, upper) = df.stat.approxQuantile(
        name,
        Array(lowerQuantile, upperQuantile),
        relativeError
      )
      col(name) >= lower && col(name) <= upper
    }.reduce(_ && _)

    df.filter(condition)
  }

  private def limitRows(df: DataFrame): DataFrame =
    df.limit(settings.rowLimit)

  private def addEnergyTier(df: DataFrame): DataFrame =
    df.withColumn(
      "energy_tier",
      when(col("energy_100g") < 150, "low")
        .when(col("energy_100g") <= 350, "medium")
        .otherwise("high")
    )
}
