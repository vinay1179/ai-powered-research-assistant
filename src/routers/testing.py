import logging
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from src.dependencies import ArxivDep, OpenSearchDep, PDFParserDep, SessionDep
from src.repositories.paper import PaperRepository
from src.schemas.api.local_ingest import (
    LocalPdfIngestItem,
    LocalPdfIngestRequest,
    LocalPdfIngestResponse,
)
from src.schemas.arxiv.paper import PaperCreate
from src.schemas.pdf_parser.models import ArxivMetadata, ParsedPaper, PdfContent
from src.services.metadata_fetcher import MetadataFetcher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/testing", tags=["testing"])


def _collect_pdf_paths(directory: str, limit: int) -> List[Path]:
    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {dir_path}")

    pdf_paths = sorted(p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    return pdf_paths[:limit]


@router.post("/ingest-local-pdfs", response_model=LocalPdfIngestResponse)
async def ingest_local_pdfs(
    request: LocalPdfIngestRequest,
    db: SessionDep,
    arxiv_client: ArxivDep,
    pdf_parser: PDFParserDep,
    opensearch_client: OpenSearchDep,
) -> LocalPdfIngestResponse:
    """
    Ingest local PDFs for testing only.
    """
    if not request.directory and not request.paths:
        raise HTTPException(status_code=400, detail="Provide either 'directory' or 'paths'")

    pdf_paths: List[Path] = []
    if request.paths:
        for path_str in request.paths:
            path = Path(path_str).expanduser().resolve()
            if not path.exists() or not path.is_file():
                raise HTTPException(status_code=400, detail=f"File not found: {path}")
            if path.suffix.lower() != ".pdf":
                raise HTTPException(status_code=400, detail=f"Not a PDF file: {path}")
            pdf_paths.append(path)
        pdf_paths = pdf_paths[: request.limit]
    elif request.directory:
        pdf_paths = _collect_pdf_paths(request.directory, request.limit)

    if not pdf_paths:
        raise HTTPException(status_code=400, detail="No PDF files found to ingest")

    metadata_fetcher = MetadataFetcher(
        arxiv_client=arxiv_client,
        pdf_parser=pdf_parser,
        opensearch_client=opensearch_client,
    )
    paper_repo = PaperRepository(db)

    results: List[LocalPdfIngestItem] = []
    for pdf_path in pdf_paths:
        item = LocalPdfIngestItem(file_path=str(pdf_path))
        try:
            arxiv_id = f"local-{pdf_path.stem}-{int(datetime.utcnow().timestamp())}"
            item.arxiv_id = arxiv_id

            parsed_content: PdfContent = await pdf_parser.parse_pdf(pdf_path)
            arxiv_metadata = ArxivMetadata(
                title=pdf_path.stem,
                authors=["Local Upload"],
                abstract="Local PDF import for testing.",
                arxiv_id=arxiv_id,
                categories=["local"],
                pdf_url=f"file://{pdf_path}",
                published_date=datetime.utcnow().isoformat(),
            )
            parsed_paper = ParsedPaper(arxiv_metadata=arxiv_metadata, pdf_content=parsed_content)
            serialized = metadata_fetcher._serialize_parsed_content(parsed_paper)

            llm_payload = {}
            if request.build_llm_context:
                llm_payload = await metadata_fetcher._build_llm_context(
                    title=arxiv_metadata.title,
                    abstract=arxiv_metadata.abstract,
                    raw_text=serialized.get("raw_text", ""),
                )

            paper_data = PaperCreate(
                arxiv_id=arxiv_id,
                title=arxiv_metadata.title,
                authors=arxiv_metadata.authors,
                abstract=arxiv_metadata.abstract,
                categories=arxiv_metadata.categories,
                pdf_url=arxiv_metadata.pdf_url,
                published_date=datetime.utcnow(),
                raw_text=serialized.get("raw_text"),
                sections=serialized.get("sections"),
                references=serialized.get("references"),
                parser_used=serialized.get("parser_used"),
                parser_metadata=serialized.get("parser_metadata"),
                pdf_processed=serialized.get("pdf_processed", True),
                pdf_processing_date=serialized.get("pdf_processing_date"),
                llm_summary=llm_payload.get("summary"),
                llm_key_points=llm_payload.get("key_points"),
                llm_context=llm_payload.get("context"),
                llm_model=llm_payload.get("model"),
                llm_generated_at=llm_payload.get("generated_at"),
            )
            paper_repo.upsert(paper_data)
            db.commit()
            item.stored = True

            if request.index_to_opensearch:
                index_payload = {
                    "arxiv_id": arxiv_id,
                    "title": arxiv_metadata.title,
                    "authors": arxiv_metadata.authors,
                    "abstract": arxiv_metadata.abstract,
                    "categories": arxiv_metadata.categories,
                    "raw_text": serialized.get("raw_text"),
                    "pdf_url": arxiv_metadata.pdf_url,
                    "published_date": arxiv_metadata.published_date,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }
                item.indexed = bool(opensearch_client.index_paper(index_payload))
        except Exception as exc:
            logger.exception("Local PDF ingestion failed for %s", pdf_path)
            db.rollback()
            item.error = str(exc)

        results.append(item)

    return LocalPdfIngestResponse(started_at=datetime.utcnow(), processed=results)
