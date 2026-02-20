import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/opt/airflow")

from src.db.factory import make_database
from src.services.indexing.factory import make_hybrid_indexing_service
from src.services.opensearch.factory import make_opensearch_client_fresh

logger = logging.getLogger(__name__)


async def _index_papers_with_chunks(papers):
    """Async helper to index papers with chunking and embeddings."""
    indexing_service = make_hybrid_indexing_service()

    papers_data = []
    for paper in papers:
        if hasattr(paper, "__dict__"):
            paper_dict = {
                "id": str(paper.id),
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "categories": paper.categories,
                "published_date": paper.published_date,
                "raw_text": paper.raw_text,
                "sections": paper.sections,
                "equations": paper.equations,
            }
        else:
            paper_dict = paper
        papers_data.append(paper_dict)

    stats = await indexing_service.index_papers_batch(papers=papers_data, replace_existing=True)
    return stats


def index_papers_hybrid(**context):
    """Index papers with chunking and vector embeddings for hybrid search."""
    try:
        database = make_database()

        ti = context.get("ti")

        fetch_results = None
        if ti:
            fetch_results = ti.xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")

        with database.get_session() as session:
            from src.models.paper import Paper

            if fetch_results and fetch_results.get("papers_stored", 0) > 0:
                from sqlalchemy import desc

                papers = session.query(Paper).order_by(desc(Paper.created_at)).limit(fetch_results["papers_stored"]).all()
            else:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=1)
                papers = session.query(Paper).filter(Paper.created_at >= cutoff_date).all()

            if not papers:
                logger.info("No papers to index for hybrid search")
                return {"papers_indexed": 0, "chunks_created": 0}

            logger.info("Indexing %s papers for hybrid search", len(papers))

            stats = asyncio.run(_index_papers_with_chunks(papers))

            logger.info(
                "Hybrid indexing complete: %s papers, %s chunks created, %s chunks indexed",
                stats["papers_processed"],
                stats["total_chunks_created"],
                stats["total_chunks_indexed"],
            )

            if ti:
                ti.xcom_push(key="hybrid_index_stats", value=stats)

            return stats

    except Exception as exc:
        logger.error("Failed to index papers for hybrid search: %s", exc)
        raise


def verify_hybrid_index(**context):
    """Verify hybrid index health and get statistics."""
    try:
        opensearch_client = make_opensearch_client_fresh()

        stats = opensearch_client.client.indices.stats(index=opensearch_client.chunk_index_name)
        count = opensearch_client.client.count(index=opensearch_client.chunk_index_name)

        paper_count_query = {"aggs": {"unique_papers": {"cardinality": {"field": "arxiv_id"}}}, "size": 0}

        paper_count_response = opensearch_client.client.search(
            index=opensearch_client.chunk_index_name, body=paper_count_query
        )

        unique_papers = paper_count_response["aggregations"]["unique_papers"]["value"]

        result = {
            "index_name": opensearch_client.chunk_index_name,
            "total_chunks": count["count"],
            "unique_papers": unique_papers,
            "avg_chunks_per_paper": (count["count"] / unique_papers if unique_papers > 0 else 0),
            "index_size_mb": stats["indices"][opensearch_client.chunk_index_name]["total"]["store"]["size_in_bytes"]
            / (1024 * 1024),
        }

        logger.info(
            "Hybrid index stats: %s chunks, %s papers, %s chunks/paper",
            result["total_chunks"],
            result["unique_papers"],
            f"{result['avg_chunks_per_paper']:.1f}",
        )

        return result

    except Exception as exc:
        logger.error("Failed to verify hybrid index: %s", exc)
        raise
