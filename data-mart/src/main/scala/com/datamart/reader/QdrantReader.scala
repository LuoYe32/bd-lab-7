package com.datamart.reader

import com.datamart.config.DataMartConfig
import com.fasterxml.jackson.databind.ObjectMapper
import okhttp3._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._


class QdrantReader(config: DataMartConfig)(implicit spark: SparkSession) {

  private val mapper: ObjectMapper = new ObjectMapper()

  private val httpClient = new OkHttpClient()
  private val jsonMediaType = MediaType.parse("application/json; charset=utf-8")

  private val baseUrl    = config.qdrant.url
  private val collection = config.qdrant.inputCollection
  private val apiKey     = config.qdrant.apiKey
  private val batchSize  = config.qdrant.scrollBatchSize
  private val columns    = config.data.selectedColumns

  def read(): DataFrame = {
    val rows = fetchAllPoints()

    if (rows.isEmpty)
      throw new IllegalStateException(
        s"No data found in Qdrant collection: $collection"
      )

    spark.createDataFrame(
      spark.sparkContext.parallelize(rows),
      buildSchema()
    )
  }

  private def fetchAllPoints(): List[Row] = {
    val allRows           = ListBuffer[Row]()
    var offset: Option[Any] = None
    var hasMore           = true

    while (hasMore) {
      val (points, nextOffset) = scrollPage(offset)

      for (point <- points) {
        val payload = point
          .get("payload")
          .asInstanceOf[java.util.Map[String, Any]]

        val values: Seq[Any] = columns.map { colName =>
          toDouble(payload.get(colName))
        }

        allRows += Row.fromSeq(values)
      }

      nextOffset match {
        case Some(_) => offset = nextOffset
        case None    => hasMore = false
      }
    }

    allRows.toList
  }

  private def scrollPage(
    offset: Option[Any]
  ): (List[java.util.Map[String, Any]], Option[Any]) = {

    val bodyMap = new java.util.HashMap[String, Any]()
    bodyMap.put("limit", batchSize)
    bodyMap.put("with_payload", true)
    bodyMap.put("with_vectors", false)
    offset.foreach(o => bodyMap.put("offset", o))

    val bodyJson = mapper.writeValueAsString(bodyMap)

    val requestBuilder = new Request.Builder()
      .url(s"$baseUrl/collections/$collection/points/scroll")
      .post(RequestBody.create(jsonMediaType, bodyJson))
      .addHeader("Content-Type", "application/json")

    if (apiKey.nonEmpty)
      requestBuilder.addHeader("api-key", apiKey)

    val response = httpClient.newCall(requestBuilder.build()).execute()

    try {
      val body   = response.body().string()
      val root   = mapper.readValue(body, classOf[java.util.Map[String, Any]])
      val result = root.get("result").asInstanceOf[java.util.Map[String, Any]]

      val points = result
        .get("points")
        .asInstanceOf[java.util.List[java.util.Map[String, Any]]]
        .asScala.toList

      val nextOffset = Option(result.get("next_page_offset"))

      (points, nextOffset)
    } finally {
      response.close()
    }
  }

  private def toDouble(value: Any): java.lang.Double =
    value match {
      case n: java.lang.Number => n.doubleValue()
      case null                => null.asInstanceOf[java.lang.Double]
      case other               => other.toString.toDouble
    }

  private def buildSchema(): StructType =
    StructType(
      columns.map(name => StructField(name, DoubleType, nullable = true))
    )
}
