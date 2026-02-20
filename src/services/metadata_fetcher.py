import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dateutil import parser as date_parser
from sqlalchemy.orm import Session
from src.config import Settings, get_settings
from src.exceptions import MetadataFetchingException, PipelineException
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import ArxivPaper, PaperCreate
from src.schemas.pdf_parser.models import ArxivMetadata, ParsedPaper, PdfContent
from src.services.arxiv.client import ArxivClient
from src.services.gemini.client import GeminiClient
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient
from src.services.pdf_parser.parser import PDFParserService

logger = logging.getLogger(__name__)


class MetadataFetcher:
    """
    Service for fetching arXiv papers with PDF processing and database storage.

    This service orchestrates the complete pipeline:
    1. Fetch paper metadata from arXiv API
    2. Download PDFs with caching
    3. Parse PDFs with Docling
    4. Store complete paper data in PostgreSQL
    """

    def __init__(
        self,
        arxiv_client: ArxivClient,
        pdf_parser: PDFParserService,
        opensearch_client: Optional[OpenSearchClient] = None,
        pdf_cache_dir: Optional[Path] = None,
        max_concurrent_downloads: int = 5,
        max_concurrent_parsing: int = 3,
        ollama_client: Optional[OllamaClient] = None,
        ollama_model: Optional[str] = None,
        ollama_context_max_chars: Optional[int] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize metadata fetcher.

        Args:
            arxiv_client: ArxivClient instance for API calls
            pdf_parser: PDFParserService for parsing PDFs
            pdf_cache_dir: Directory for PDF caching (uses client default if None)
            max_concurrent_downloads: Maximum concurrent PDF downloads
            max_concurrent_parsing: Maximum concurrent PDF parsing operations
        """
        self.arxiv_client = arxiv_client
        self.pdf_parser = pdf_parser
        self.opensearch_client = opensearch_client
        self.pdf_cache_dir = pdf_cache_dir or self.arxiv_client.pdf_cache_dir
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_parsing = max_concurrent_parsing
        self.settings = settings or get_settings()
        self.ollama_client = ollama_client or OllamaClient(self.settings)
        self.gemini_client = GeminiClient(self.settings)
        self.ollama_model = ollama_model or self.settings.ollama_default_model
        self.ollama_context_max_chars = ollama_context_max_chars or self.settings.ollama_context_max_chars
        self.llm_provider = self.settings.llm_provider

    async def fetch_and_process_papers(
        self,
        max_results: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        process_pdfs: bool = True,
        build_ollama_context: bool = True,
        store_to_db: bool = True,
        db_session: Optional[Session] = None,
        index_to_opensearch: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch papers from arXiv, process PDFs, and store to database.

        Args:
            max_results: Maximum papers to fetch
            from_date: Filter papers from this date (YYYYMMDD)
            to_date: Filter papers to this date (YYYYMMDD)
            process_pdfs: Whether to download and parse PDFs
            store_to_db: Whether to store results in database
            db_session: Database session (required if store_to_db=True)

        Returns:
            Dictionary with processing results and statistics
        """

        results = {
            "papers_fetched": 0,
            "pdfs_downloaded": 0,
            "pdfs_parsed": 0,
            "papers_stored": 0,
            "papers_indexed": 0,
            "errors": [],
            "processing_time": 0,
        }

        start_time = datetime.now()

        try:
            # Step 1: Fetch paper metadata from arXiv
            papers = await self.arxiv_client.fetch_papers(
                max_results=max_results, from_date=from_date, to_date=to_date, sort_by="submittedDate", sort_order="descending"
            )

            results["papers_fetched"] = len(papers)

            if not papers:
                logger.warning("No papers found")
                return results

            # Step 2: Process PDFs if requested
            pdf_results = {}
            if process_pdfs:
                pdf_results = await self._process_pdfs_batch(papers, build_ollama_context=build_ollama_context)
                results["pdfs_downloaded"] = pdf_results["downloaded"]
                results["pdfs_parsed"] = pdf_results["parsed"]
                results["errors"].extend(pdf_results["errors"])

            # Step 3: Store to database if requested
            if store_to_db and db_session:
                logger.info("Step 3: Storing papers to database...")
                stored_count = self._store_papers_to_db(papers, pdf_results.get("parsed_papers", {}), db_session)
                results["papers_stored"] = stored_count
            elif store_to_db:
                logger.warning("Database storage requested but no session provided")
                results["errors"].append("Database session not provided for storage")

            # Step 4: Index to OpenSearch if requested
            if index_to_opensearch and self.opensearch_client:
                logger.info("Step 4: Indexing papers to OpenSearch...")
                indexed_count = self._index_papers_to_opensearch(papers, pdf_results.get("parsed_papers", {}))
                results["papers_indexed"] = indexed_count
            elif index_to_opensearch and not self.opensearch_client:
                logger.warning("OpenSearch indexing requested but no client provided")
                results["errors"].append("OpenSearch client not provided for indexing")

            # Calculate total processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            results["processing_time"] = processing_time

            # Simple logging summary
            logger.info(
                f"Pipeline completed in {processing_time:.1f}s: {results['papers_fetched']} papers, {results['pdfs_downloaded']} PDFs, {len(results['errors'])} errors"
            )

            if results["errors"]:
                logger.warning("Errors summary:")
                for i, error in enumerate(results["errors"][:5], 1):  # Show first 5 errors
                    logger.warning(f"  {i}. {error}")
                if len(results["errors"]) > 5:
                    logger.warning(f"  ... and {len(results['errors']) - 5} more errors")

            return results

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            results["errors"].append(f"Pipeline error: {str(e)}")
            raise PipelineException(f"Pipeline execution failed: {e}") from e

    async def _process_pdfs_batch(self, papers: List[ArxivPaper], build_ollama_context: bool) -> Dict[str, Any]:
        """
        Process PDFs for a batch of papers with async concurrency.

        Uses overlapping download+parse pipeline:
        - Downloads happen concurrently (up to max_concurrent_downloads)
        - As each download completes, parsing starts immediately
        - Multiple PDFs can be parsing while others are still downloading

        This is optimal for production workloads like 100 papers/day.

        Args:
            papers: List of ArxivPaper objects

        Returns:
            Dictionary with processing results and statistics
        """
        results = {
            "downloaded": 0,
            "parsed": 0,
            "parsed_papers": {},
            "errors": [],
            "download_failures": [],
            "parse_failures": [],
        }

        logger.info(f"Starting async pipeline for {len(papers)} PDFs...")
        logger.info(f"Concurrent downloads: {self.max_concurrent_downloads}")
        logger.info(f"Concurrent parsing: {self.max_concurrent_parsing}")

        # Create semaphores for controlled concurrency
        download_semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        parse_semaphore = asyncio.Semaphore(self.max_concurrent_parsing)

        # Start all download+parse pipelines concurrently
        pipeline_tasks = [
            self._download_and_parse_pipeline(paper, download_semaphore, parse_semaphore, build_ollama_context)
            for paper in papers
        ]

        # Wait for all pipelines to complete
        pipeline_results = await asyncio.gather(*pipeline_tasks, return_exceptions=True)

        # Process results with detailed error tracking
        for paper, result in zip(papers, pipeline_results):
            if isinstance(result, Exception):
                error_msg = f"Pipeline error for {paper.arxiv_id}: {str(result)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
            elif result:
                # Result is tuple: (download_success, parsed_paper)
                download_success, parsed_paper = result

                if download_success:
                    results["downloaded"] += 1

                    if parsed_paper:
                        results["parsed"] += 1
                        results["parsed_papers"][paper.arxiv_id] = parsed_paper
                    else:
                        # Download succeeded but parsing failed
                        results["parse_failures"].append(paper.arxiv_id)
                else:
                    # Download failed
                    results["download_failures"].append(paper.arxiv_id)
            else:
                # No result returned (shouldn't happen but handle gracefully)
                results["download_failures"].append(paper.arxiv_id)

        # Simple processing summary
        logger.info(f"PDF processing: {results['downloaded']}/{len(papers)} downloaded, {results['parsed']} parsed")

        if results["download_failures"]:
            logger.warning(f"Download failures: {len(results['download_failures'])}")

        if results["parse_failures"]:
            logger.warning(f"Parse failures: {len(results['parse_failures'])}")

        # Add specific failure info to general errors list for backward compatibility
        if results["download_failures"]:
            results["errors"].extend([f"Download failed: {arxiv_id}" for arxiv_id in results["download_failures"]])
        if results["parse_failures"]:
            results["errors"].extend([f"PDF parse failed: {arxiv_id}" for arxiv_id in results["parse_failures"]])

        return results

    async def _download_and_parse_pipeline(
        self,
        paper: ArxivPaper,
        download_semaphore: asyncio.Semaphore,
        parse_semaphore: asyncio.Semaphore,
        build_ollama_context: bool,
    ) -> tuple:
        """
        Complete download+parse pipeline for a single paper with true parallelism.
        Downloads PDF, then immediately starts parsing while other downloads continue.

        Returns:
            Tuple of (download_success: bool, parsed_paper: Optional[ParsedPaper])
        """
        download_success = False
        parsed_paper = None

        try:
            # Step 1: Download PDF with download concurrency control
            async with download_semaphore:
                logger.debug(f"Starting download: {paper.arxiv_id}")
                pdf_path = await self.arxiv_client.download_pdf(paper, False)

                if pdf_path:
                    download_success = True
                    logger.debug(f"Download complete: {paper.arxiv_id}")
                else:
                    logger.error(f"Download failed: {paper.arxiv_id}")
                    return (False, None)

            # Step 2: Parse PDF with parse concurrency control (happens AFTER download completes)
            # This allows other downloads to continue while this PDF is being parsed
            async with parse_semaphore:
                logger.debug(f"Starting parse: {paper.arxiv_id}")
                pdf_content = await self.pdf_parser.parse_pdf(pdf_path)

                if pdf_content:
                    # Create ArxivMetadata from the paper
                    arxiv_metadata = ArxivMetadata(
                        title=paper.title,
                        authors=paper.authors,
                        abstract=paper.abstract,
                        arxiv_id=paper.arxiv_id,
                        categories=paper.categories,
                        published_date=paper.published_date,
                        pdf_url=paper.pdf_url,
                    )

                    # Combine into ParsedPaper
                    parsed_paper = ParsedPaper(arxiv_metadata=arxiv_metadata, pdf_content=pdf_content)

                    if build_ollama_context:
                        llm_payload = await self._build_llm_context(
                            title=paper.title,
                            abstract=paper.abstract,
                            raw_text=pdf_content.raw_text,
                        )
                        parsed_paper.llm_summary = llm_payload.get("summary")
                        parsed_paper.llm_key_points = llm_payload.get("key_points")
                        parsed_paper.llm_context = llm_payload.get("context")
                        parsed_paper.llm_model = llm_payload.get("model")
                        parsed_paper.llm_generated_at = llm_payload.get("generated_at")

                    if pdf_content.equations:
                        await self._build_equation_explanations(
                            equations=pdf_content.equations,
                            sections=pdf_content.sections,
                        )
                    logger.debug(f"Parse complete: {paper.arxiv_id} - {len(pdf_content.raw_text)} chars extracted")
                else:
                    # PDF parsing failed, but this is not critical - we can continue with metadata only
                    logger.warning(f"PDF parsing failed for {paper.arxiv_id}, continuing with metadata only")

        except Exception as e:
            logger.error(f"Pipeline error for {paper.arxiv_id}: {e}")
            raise MetadataFetchingException(f"Pipeline error for {paper.arxiv_id}: {e}") from e

        return (download_success, parsed_paper)

    async def _build_llm_context(self, title: str, abstract: str, raw_text: str) -> Dict[str, Any]:
        prompt_text = (
            "You are a research assistant. Summarize the paper, list key points, and provide a short context for Q&A.\n"
            "Return ONLY valid JSON. Do not include markdown, headings, or any extra text.\n\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n\n"
            "Paper content (truncated):\n"
            f"{raw_text[: self.ollama_context_max_chars]}\n\n"
            'Required JSON shape: {"summary": "...", "key_points": ["..."], "context": "..."}'
        )

        try:
            if self.llm_provider == "gemini":
                response = await self.gemini_client.generate(prompt_text, response_mime_type="application/json")
                raw_response = None
                if isinstance(response, dict):
                    candidates = response.get("candidates") or []
                    if candidates:
                        content = candidates[0].get("content") or {}
                        parts = content.get("parts") or []
                        if parts:
                            raw_response = parts[0].get("text")
            else:
                response = await self.ollama_client.generate(self.ollama_model, prompt_text, format="json")
                raw_response = response.get("response") if isinstance(response, dict) else None

            if not raw_response:
                return {}

            try:
                import json

                cleaned = raw_response.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    cleaned = cleaned.replace("json", "", 1).strip()
                try:
                    parsed = json.loads(cleaned)
                except Exception:
                    start = cleaned.find("{")
                    end = cleaned.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        parsed = json.loads(cleaned[start : end + 1])
                    else:
                        raise
                return {
                    "summary": parsed.get("summary"),
                    "key_points": parsed.get("key_points"),
                    "context": parsed.get("context"),
                    "model": self.settings.gemini_model if self.llm_provider == "gemini" else self.ollama_model,
                    "generated_at": datetime.now(),
                }
            except Exception:
                return {
                    "summary": None,
                    "key_points": None,
                    "context": raw_response,
                    "model": self.settings.gemini_model if self.llm_provider == "gemini" else self.ollama_model,
                    "generated_at": datetime.now(),
                }
        except Exception as exc:
            logger.warning("Failed to build LLM context: %s", exc)
            return {}

    async def _build_equation_explanations(self, equations: List[Any], sections: List[Any]) -> None:
        section_map = {section.title: section.content for section in sections}
        for equation in equations:
            context_text = section_map.get(equation.section_title, "")
            context_snippet = context_text[:1000] if context_text else ""
            prompt_text = (
                "You are a research assistant. Explain the following LaTeX equation in plain English.\n"
                "Use the surrounding section context to interpret symbols and meaning. Keep it concise (1-3 sentences).\n\n"
                f"Section title: {equation.section_title or 'Unknown'}\n"
                f"Section context (truncated):\n{context_snippet}\n\n"
                f"LaTeX equation:\n{equation.latex}\n"
            )
            try:
                response = await self.gemini_client.generate(prompt_text)
                explanation = ""
                if isinstance(response, dict):
                    explanation = GeminiClient._extract_text(response) or ""
                equation.explanation = explanation.strip() or "Explanation unavailable."
            except Exception as exc:
                logger.warning("Failed to build equation explanation: %s", exc)
                equation.explanation = "Explanation unavailable."

    def _serialize_parsed_content(self, parsed_paper: ParsedPaper) -> Dict[str, Any]:
        """
        Serialize ParsedPaper content for database storage.

        Args:
            parsed_paper: ParsedPaper object with PDF content

        Returns:
            Dictionary with serialized content for database storage
        """
        try:
            pdf_content = parsed_paper.pdf_content

            # Serialize sections
            sections = [{"title": section.title, "content": section.content} for section in pdf_content.sections]

            # Serialize references
            references = list(pdf_content.references)  #

            # Serialize equations
            equations = [
                {
                    "latex": equation.latex,
                    "explanation": equation.explanation,
                    "section_title": equation.section_title,
                    "block_order": equation.block_order,
                }
                for equation in pdf_content.equations
            ]

            return {
                "raw_text": pdf_content.raw_text,
                "sections": sections,
                "references": references,
                "equations": equations,
                "parser_used": pdf_content.parser_used.value if pdf_content.parser_used else None,
                "parser_metadata": pdf_content.metadata or {},
                "pdf_processed": True,
                "pdf_processing_date": datetime.now(),
                "llm_summary": parsed_paper.llm_summary,
                "llm_key_points": parsed_paper.llm_key_points,
                "llm_context": parsed_paper.llm_context,
                "llm_model": parsed_paper.llm_model,
                "llm_generated_at": parsed_paper.llm_generated_at,
            }
        except Exception as e:
            logger.error(f"Failed to serialize parsed content: {e}")
            return {"pdf_processed": False, "parser_metadata": {"error": str(e)}}

    def _store_papers_to_db(
        self,
        papers: List[ArxivPaper],
        parsed_papers: Dict[str, ParsedPaper],
        db_session: Session,
    ) -> int:
        """
        Store papers and parsed content to database with comprehensive content storage.

        Args:
            papers: List of ArxivPaper metadata
            parsed_papers: Dictionary of parsed PDF content by arxiv_id
            db_session: Database session

        Returns:
            Number of papers stored successfully
        """
        paper_repo = PaperRepository(db_session)
        stored_count = 0

        for paper in papers:
            try:
                # Get parsed content if available
                parsed_paper = parsed_papers.get(paper.arxiv_id)

                # Base paper data
                published_date = (
                    date_parser.parse(paper.published_date) if isinstance(paper.published_date, str) else paper.published_date
                )
                paper_data = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "published_date": published_date,
                    "pdf_url": paper.pdf_url,
                }

                # Add parsed content if available
                if parsed_paper:
                    parsed_content = self._serialize_parsed_content(parsed_paper)
                    paper_data.update(parsed_content)
                    logger.debug(
                        f"Storing paper {paper.arxiv_id} with parsed content ({len(parsed_content.get('raw_text', '')) if parsed_content.get('raw_text') else 0} chars)"
                    )
                else:
                    # No parsed content - just store metadata
                    paper_data.update(
                        {"pdf_processed": False, "parser_metadata": {"note": "PDF processing not available or failed"}}
                    )
                    logger.debug(f"Storing paper {paper.arxiv_id} with metadata only")

                paper_create = PaperCreate(**paper_data)
                stored_paper = paper_repo.upsert(paper_create)

                if stored_paper:
                    stored_count += 1
                    content_info = "with parsed content" if parsed_paper else "metadata only"
                    logger.debug(f"Stored paper {paper.arxiv_id} to database ({content_info})")

            except Exception as e:
                logger.error(f"Failed to store paper {paper.arxiv_id}: {e}")

        # Commit all changes
        try:
            db_session.commit()
            logger.info(f"Committed {stored_count} papers to database with full content storage")
        except Exception as e:
            logger.error(f"Failed to commit papers to database: {e}")
            db_session.rollback()
            stored_count = 0

        return stored_count

    def _index_papers_to_opensearch(
        self,
        papers: List[ArxivPaper],
        parsed_papers: Dict[str, ParsedPaper],
    ) -> int:
        """
        Index papers to OpenSearch for full-text search.

        Args:
            papers: List of ArxivPaper metadata
            parsed_papers: Dictionary of parsed PDF content by arxiv_id

        Returns:
            Number of papers successfully indexed
        """
        indexed_count = 0

        for paper in papers:
            try:
                # Get parsed content if available
                parsed_paper = parsed_papers.get(paper.arxiv_id)

                # Prepare data for OpenSearch
                opensearch_data = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors if isinstance(paper.authors, str) else ", ".join(paper.authors),
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                    "published_date": paper.published_date.isoformat()
                    if hasattr(paper.published_date, "isoformat")
                    else str(paper.published_date),
                }

                # Add parsed content if available
                if parsed_paper and parsed_paper.pdf_content:
                    max_text_size = self.settings.opensearch.max_text_size
                    opensearch_data["raw_text"] = parsed_paper.pdf_content.raw_text[:max_text_size]
                else:
                    opensearch_data["raw_text"] = ""

                # Index to OpenSearch
                if self.opensearch_client.index_paper(opensearch_data):
                    indexed_count += 1
                    logger.debug(f"Indexed paper {paper.arxiv_id} to OpenSearch")
                else:
                    logger.warning(f"Failed to index paper {paper.arxiv_id} to OpenSearch")

            except Exception as e:
                logger.error(f"Error indexing paper {paper.arxiv_id} to OpenSearch: {e}")

        logger.info(f"Indexed {indexed_count}/{len(papers)} papers to OpenSearch")
        return indexed_count


def make_metadata_fetcher(
    arxiv_client: ArxivClient,
    pdf_parser: PDFParserService,
    opensearch_client: Optional[OpenSearchClient] = None,
    pdf_cache_dir: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> MetadataFetcher:
    """
    Factory function to create MetadataFetcher instance optimized for production.

    Configured for typical production workloads (100 papers/day):
    - 5 concurrent downloads (I/O bound, can handle more)
    - 3 concurrent parsing operations (CPU intensive, use fewer)
    - Async pipeline for optimal resource utilization

    Args:
        arxiv_client: Configured ArxivClient
        pdf_parser: Configured PDFParserService (singleton with model caching)
        pdf_cache_dir: Optional PDF cache directory

    Returns:
        MetadataFetcher instance optimized for production
    """
    if settings is None:
        settings = get_settings()

    return MetadataFetcher(
        arxiv_client=arxiv_client,
        pdf_parser=pdf_parser,
        opensearch_client=opensearch_client,
        pdf_cache_dir=pdf_cache_dir,
        max_concurrent_downloads=settings.arxiv.max_concurrent_downloads,
        max_concurrent_parsing=settings.arxiv.max_concurrent_parsing,
        ollama_client=None,
        settings=settings,
    )
