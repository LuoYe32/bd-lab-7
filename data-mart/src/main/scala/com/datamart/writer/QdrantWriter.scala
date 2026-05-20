package com.datamart.writer

import com.datamart.config.DataMartConfig
import com.fasterxml.jackson.databind.ObjectMapper
import okhttp3._
import org.apache.spark.sql.DataFrame

import scala.collection.JavaConverters._


class QdrantWriter(config: DataMartConfig, collectionName: String) {

  private val mapper: ObjectMapper = new ObjectMapper()

  private val httpClient    = new OkHttpClient()
  private val jsonMediaType = MediaType.parse("application/json; charset=utf-8")

  private val baseUrl   = config.qdrant.url
  private val apiKey    = config.qdrant.apiKey
  private val batchSize = config.qdrant.uploadBatchSize
  private val vectorSize = config.qdrant.vectorSize
  private val columns   = config.data.selectedColumns


  def recreateCollection(): Unit = {
    if (collectionExists()) {
      println(s"[QdrantWriter] Deleting existing collection '$collectionName'...")
      deleteCollection()
    }
    println(s"[QdrantWriter] Creating collection '$collectionName' (size=$vectorSize, distance=Cosine)...")
    createCollection()
  }

  def write(df: DataFrame): Unit = {
    val rows = df.collect()
    println(s"[QdrantWriter] Uploading ${rows.length} rows to '$collectionName'...")

    rows.zipWithIndex
      .grouped(batchSize)
      .foreach { batch =>
        upsertBatch(batch.map { case (row, idx) => buildPoint(row, idx) })
      }

    println(s"[QdrantWriter] Upload complete.")
  }

  def writeWithCluster(df: DataFrame, predictionCol: String): Unit = {
    val rows = df.collect()
    println(s"[QdrantWriter] Uploading ${rows.length} clustered rows to '$collectionName'...")

    rows.zipWithIndex
      .grouped(batchSize)
      .foreach { batch =>
        upsertBatch(batch.map { case (row, idx) => buildClusteredPoint(row, idx, predictionCol) })
      }

    println(s"[QdrantWriter] Upload complete.")
  }

  private def buildPoint(row: org.apache.spark.sql.Row, id: Int): java.util.Map[String, Any] = {
    val vector: java.util.List[Double] = columns.map(row.getAs[Double](_)).asJava

    val payload = new java.util.HashMap[String, Any]()
    columns.foreach(c => payload.put(c, row.getAs[Double](c)))

    if (row.schema.fieldNames.contains("energy_tier"))
      payload.put("energy_tier", row.getAs[String]("energy_tier"))

    val point = new java.util.HashMap[String, Any]()
    point.put("id", id)
    point.put("vector", vector)
    point.put("payload", payload)
    point
  }

  private def buildClusteredPoint(
    row: org.apache.spark.sql.Row,
    id: Int,
    predictionCol: String
  ): java.util.Map[String, Any] = {
    val point = buildPoint(row, id)
    val payload = point.get("payload").asInstanceOf[java.util.HashMap[String, Any]]
    payload.put("cluster", row.getAs[Int](predictionCol).asInstanceOf[Any])
    point
  }

  private def upsertBatch(points: Array[java.util.Map[String, Any]]): Unit = {
    val body = new java.util.HashMap[String, Any]()
    body.put("points", points.toList.asJava)
    sendRequest("PUT", s"/collections/$collectionName/points", payload = Some(body))
  }

  private def collectionExists(): Boolean = {
    val request = buildRequest("GET", s"/collections/$collectionName", payload = None)
    val response = httpClient.newCall(request).execute()
    try { response.code() == 200 } finally { response.close() }
  }

  private def deleteCollection(): Unit =
    sendRequest("DELETE", s"/collections/$collectionName", payload = None)

  private def createCollection(): Unit = {
    val vectorParams = new java.util.HashMap[String, Any]()
    vectorParams.put("size", vectorSize)
    vectorParams.put("distance", "Cosine")

    val body = new java.util.HashMap[String, Any]()
    body.put("vectors", vectorParams)

    sendRequest("PUT", s"/collections/$collectionName", payload = Some(body))
  }

  private def sendRequest(method: String, path: String, payload: Option[java.util.Map[String, Any]]): Unit = {
    val response = httpClient.newCall(buildRequest(method, path, payload)).execute()
    try {
      if (!response.isSuccessful)
        throw new RuntimeException(
          s"Qdrant $method $path failed: ${response.code()} ${response.body().string()}"
        )
    } finally { response.close() }
  }

  private def buildRequest(method: String, path: String, payload: Option[java.util.Map[String, Any]]): Request = {
    val bodyOrEmpty: RequestBody = payload match {
      case Some(p) => RequestBody.create(jsonMediaType, mapper.writeValueAsString(p))
      case None    => RequestBody.create(jsonMediaType, "")
    }

    val reqBody: RequestBody = method match {
      case "GET"    => null
      case "DELETE" => RequestBody.create(jsonMediaType, "")
      case _        => bodyOrEmpty
    }

    val builder = new Request.Builder()
      .url(s"$baseUrl$path")
      .method(method, reqBody)
      .addHeader("Content-Type", "application/json")

    if (apiKey.nonEmpty) builder.addHeader("api-key", apiKey)

    builder.build()
  }
}
