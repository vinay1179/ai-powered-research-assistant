from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for RAG question answering."""

    query: str = Field(..., description="User's question", min_length=1, max_length=1000)
    top_k: int = Field(3, description="Number of top chunks to retrieve", ge=1, le=10)
    use_hybrid: bool = Field(True, description="Use hybrid search (BM25 + vector)")
    model: str = Field("llama3.2:1b", description="Ollama model to use for generation")
    provider: Literal["ollama", "gemini"] = Field("ollama", description="LLM provider to use")
    categories: Optional[List[str]] = Field(None, description="Filter by arXiv categories")


class AskResponse(BaseModel):
    """Response model for RAG question answering."""

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer from LLM")
    sources: List[str] = Field(..., description="PDF URLs of source papers")
    chunks_used: int = Field(..., description="Number of chunks used for generation")
    search_mode: str = Field(..., description="Search mode used: bm25 or hybrid")


class AgenticAskResponse(AskResponse):
    """Response model for agentic RAG question answering."""

    reasoning_steps: List[str] = Field(..., description="Agent decision steps")
    retrieval_attempts: int = Field(..., description="Number of retrieval attempts")
    trace_id: Optional[str] = Field(None, description="Langfuse trace ID for feedback")


class FeedbackRequest(BaseModel):
    """Request model for user feedback on agentic answers."""

    trace_id: str = Field(..., description="Langfuse trace ID from the response")
    score: float = Field(..., description="Feedback score (0-1 or -1 to 1)", ge=-1, le=1)
    comment: Optional[str] = Field(None, description="Optional feedback comment", max_length=1000)


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""

    success: bool = Field(..., description="Whether feedback was recorded successfully")
    message: str = Field(..., description="Status message")
