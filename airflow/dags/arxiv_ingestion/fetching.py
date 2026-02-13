import asyncio
import logging
from datetime import datetime, timedelta

from .common import get_cached_services

logger = logging.getLogger(__name__)


async def run_paper_ingestion_pipeline(target_date: str, process_pdfs: bool = True) -> dict:
    """Async wrapper for the paper ingestion pipeline."""
    arxiv_client, _, database, metadata_fetcher, _ = get_cached_services()

    max_results = arxiv_client.max_results
    logger.info("Using default max_results from config: %s", max_results)

    with database.get_session() as session:
        return await metadata_fetcher.fetch_and_process_papers(
            max_results=max_results,
            from_date=target_date,
            to_date=target_date,
            process_pdfs=process_pdfs,
            store_to_db=True,
            db_session=session,
            index_to_opensearch=False,
        )


def fetch_daily_papers(**context):
    """Fetch daily papers from arXiv and store in PostgreSQL."""
    logger.info("Starting daily paper fetching task")

    execution_date = context.get("execution_date")
    if execution_date:
        target_dt = execution_date - timedelta(days=1)
        target_date = target_dt.strftime("%Y%m%d")
    else:
        yesterday = datetime.now() - timedelta(days=1)
        target_date = yesterday.strftime("%Y%m%d")

    logger.info("Fetching papers for date: %s", target_date)

    results = asyncio.run(
        run_paper_ingestion_pipeline(
            target_date=target_date,
            process_pdfs=True,
        )
    )

    logger.info("Daily fetch complete: %s papers for %s", results.get("papers_fetched", 0), target_date)

    results["date"] = target_date
    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="fetch_results", value=results)

    return results
