from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class LocalPdfIngestRequest(BaseModel):
    """Request model for local PDF ingestion (testing only)."""

    directory: Optional[str] = Field(
        default=None, description="Directory containing PDFs to ingest (optional if paths provided)"
    )
    paths: Optional[List[str]] = Field(default=None, description="Explicit list of PDF file paths to ingest")
    limit: int = Field(default=2, ge=1, le=5, description="Maximum number of PDFs to ingest")
    build_llm_context: bool = Field(default=True, description="Whether to build LLM context")
    index_to_opensearch: bool = Field(default=True, description="Whether to index to OpenSearch")


class LocalPdfIngestItem(BaseModel):
    """Result for a single local PDF ingestion."""

    file_path: str
    arxiv_id: Optional[str] = None
    stored: bool = False
    indexed: bool = False
    error: Optional[str] = None


class LocalPdfIngestResponse(BaseModel):
    """Response for local PDF ingestion."""

    started_at: datetime
    processed: List[LocalPdfIngestItem]
