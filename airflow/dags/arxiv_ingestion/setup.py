import logging

from sqlalchemy import text

from .common import get_cached_services

logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment and verify dependencies.

    Creates hybrid search index with RRF pipeline.
    """
    logger.info("Setting up environment for arXiv paper ingestion")

    try:
        arxiv_client, _pdf_parser, database, _metadata_fetcher, opensearch_client = get_cached_services()

        with database.get_session() as session:
            session.execute(text("SELECT 1"))
            logger.info("Database connection verified")

        try:
            health = opensearch_client.client.cluster.health()
            if health["status"] in ["green", "yellow", "red"]:
                logger.info("OpenSearch hybrid client connected (cluster status: %s)", health["status"])
            else:
                raise Exception(f"OpenSearch cluster unhealthy: {health['status']}")
        except Exception as exc:
            raise Exception(f"OpenSearch hybrid client connection failed: {exc}")

        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Hybrid search index created with vector support")
        else:
            logger.info("Hybrid search index already exists")

        if setup_results.get("rrf_pipeline"):
            logger.info("RRF pipeline created successfully")
        else:
            logger.info("RRF pipeline already exists")

        logger.info("Hybrid search setup completed")

        logger.info("arXiv client ready: %s", arxiv_client.base_url)
        logger.info("PDF parser service ready (Docling models cached)")

        return {"status": "success", "message": "Environment setup completed"}

    except Exception as exc:
        error_msg = f"Environment setup failed: {str(exc)}"
        logger.error(error_msg)
        raise Exception(error_msg)
