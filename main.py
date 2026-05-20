import os
import subprocess

from qdrant_client import QdrantClient

from src.config.config import ProjectConfig
from src.pipeline.clustering_pipeline import ClusteringPipeline
from src.utils.logger import LoggerFactory


def collection_has_data(client: QdrantClient, collection_name: str) -> bool:
    if not client.collection_exists(collection_name):
        return False
    return client.get_collection(collection_name).points_count > 0


def run_data_mart(config: ProjectConfig, mode: str = "preprocess") -> None:
    logger = LoggerFactory.get_logger("DataMart")

    jar = os.path.join(os.path.dirname(__file__), config.data_mart.jar_path)

    if not os.path.exists(jar):
        logger.info("JAR not found, building with sbt assembly...")
        data_mart_dir = os.path.join(os.path.dirname(__file__), "data-mart")
        subprocess.run(["sbt", "assembly"], cwd=data_mart_dir, check=True)
        logger.info("sbt assembly complete.")

    logger.info("Running Scala Data Mart in '%s' mode...", mode)

    env = {
        **os.environ,
        "SPARK_HOME": config.data_mart.resolved_spark_home,
        "QDRANT_URL": config.qdrant.url,
        "QDRANT_API_KEY": config.qdrant.api_key,
        "QDRANT_INPUT_COLLECTION": config.qdrant.input_collection,
        "QDRANT_OUTPUT_COLLECTION": config.qdrant.output_collection,
        "DATA_PATH": os.path.join(os.path.dirname(__file__), config.data.data_path),
        "PREPROCESSED_PATH": os.path.join(os.path.dirname(__file__), config.data.preprocessed_path),
        "PREDICTIONS_PATH": os.path.join(os.path.dirname(__file__), config.data.predictions_path),
    }

    result = subprocess.run(
        [
            "spark-submit",
            "--driver-memory", "2g",
            "--driver-java-options", "-Djava.security.manager=allow",
            "--class", "com.datamart.DataMartApp",
            jar,
            mode,
        ],
        env=env,
        check=True,
    )

    logger.info("Data Mart '%s' finished (exit code %s).", mode, result.returncode)


def main() -> None:
    logger = LoggerFactory.get_logger("Main")
    config = ProjectConfig()

    client = QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)
    if not collection_has_data(client, config.qdrant.input_collection):
        logger.info("'%s' is empty — uploading raw data via Data Mart...", config.qdrant.input_collection)
        run_data_mart(config, mode="upload")
    else:
        logger.info("'%s' already has data.", config.qdrant.input_collection)

    logger.info("Running Data Mart preprocessing...")
    run_data_mart(config, mode="preprocess")

    logger.info("Running clustering pipeline...")
    ClusteringPipeline().run()

    logger.info("Writing clustering results via Data Mart...")
    run_data_mart(config, mode="write-results")


if __name__ == "__main__":
    main()
