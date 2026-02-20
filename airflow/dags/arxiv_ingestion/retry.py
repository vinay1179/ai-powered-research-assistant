import asyncio
import logging
import sys

from sqlalchemy import text

sys.path.insert(0, "/opt/airflow")

from src.schemas.arxiv.paper import ArxivPaper
from src.schemas.pdf_parser.models import ArxivMetadata, ParsedPaper
from src.services.indexing.factory import make_hybrid_indexing_service

from .common import get_cached_services

logger = logging.getLogger(__name__)


def process_failed_pdfs(**context):
    """Retry processing PDFs that failed during the main fetch task."""
    logger.info("Retrying failed PDFs (hybrid indexing)")

    fetch_results = context["task_instance"].xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")
    papers_stored = fetch_results.get("papers_stored", 0) if fetch_results else 0
    limit_hint = papers_stored if papers_stored > 0 else 100

    _arxiv_client, pdf_parser, database, metadata_fetcher, _opensearch_client = get_cached_services()

    retry_results = {
        "status": "completed",
        "papers_retried": 0,
        "papers_fixed": 0,
        "papers_failed": 0,
        "papers_llm_context_built": 0,
        "papers_reindexed": 0,
        "papers_reindex_failed": 0,
    }

    async def retry_unprocessed(papers):
        indexing_service = make_hybrid_indexing_service()
        for paper in papers:
            retry_results["papers_retried"] += 1
            try:
                arxiv_paper = ArxivPaper(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    authors=paper.authors if isinstance(paper.authors, list) else [str(paper.authors)],
                    abstract=paper.abstract,
                    categories=paper.categories or [],
                    published_date=paper.published_date.isoformat()
                    if hasattr(paper.published_date, "isoformat")
                    else str(paper.published_date),
                    pdf_url=paper.pdf_url,
                )

                pdf_path = await _arxiv_client.download_pdf(arxiv_paper, force_download=False)
                if not pdf_path:
                    retry_results["papers_failed"] += 1
                    continue

                pdf_content = await pdf_parser.parse_pdf(pdf_path)
                arxiv_metadata = ArxivMetadata(
                    title=arxiv_paper.title,
                    authors=arxiv_paper.authors,
                    abstract=arxiv_paper.abstract,
                    arxiv_id=arxiv_paper.arxiv_id,
                    categories=arxiv_paper.categories,
                    published_date=arxiv_paper.published_date,
                    pdf_url=arxiv_paper.pdf_url,
                )
                parsed_paper = ParsedPaper(arxiv_metadata=arxiv_metadata, pdf_content=pdf_content)
                parsed_fields = metadata_fetcher._serialize_parsed_content(parsed_paper)

                llm_payload = await metadata_fetcher._build_llm_context(
                    title=arxiv_paper.title,
                    abstract=arxiv_paper.abstract,
                    raw_text=pdf_content.raw_text,
                )
                if llm_payload:
                    parsed_fields.update(
                        {
                            "llm_summary": llm_payload.get("summary"),
                            "llm_key_points": llm_payload.get("key_points"),
                            "llm_context": llm_payload.get("context"),
                            "llm_model": llm_payload.get("model"),
                            "llm_generated_at": llm_payload.get("generated_at"),
                        }
                    )
                    retry_results["papers_llm_context_built"] += 1

                for key, value in parsed_fields.items():
                    setattr(paper, key, value)

                paper_data = {
                    "id": str(paper.id),
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "published_date": paper.published_date,
                    "raw_text": parsed_fields.get("raw_text", ""),
                    "sections": parsed_fields.get("sections"),
                }

                index_stats = await indexing_service.index_paper(paper_data)
                if index_stats.get("chunks_indexed", 0) > 0:
                    retry_results["papers_reindexed"] += 1
                else:
                    retry_results["papers_reindex_failed"] += 1

                retry_results["papers_fixed"] += 1
            except Exception as exc:
                logger.warning("Retry failed for %s: %s", paper.arxiv_id, exc)
                retry_results["papers_failed"] += 1

    with database.get_session() as session:
        from src.repositories.paper import PaperRepository

        paper_repo = PaperRepository(session)
        query = f"""
            SELECT * FROM papers
            WHERE DATE(created_at) = CURRENT_DATE
            AND (pdf_processed = false OR pdf_processed IS NULL)
            ORDER BY created_at DESC
            LIMIT {limit_hint}
        """
        result = session.execute(text(query))
        rows = result.fetchall()
        papers = [paper_repo.get_by_id(row.id) for row in rows]
        papers = [paper for paper in papers if paper]

        if not papers:
            logger.info("No unprocessed papers found for retry")
            return {"status": "skipped", "message": "No unprocessed papers found"}

        asyncio.run(retry_unprocessed(papers))
        session.commit()

    return retry_results
