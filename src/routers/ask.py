from fastapi import APIRouter
from src.schemas.ask import AskRequest, AskResponse, PaperSource

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Mock implementation for question answering endpoint.

    Week 1: Returns hardcoded mock data for testing.
    """
    # Mock response only
    mock_sources = [
        PaperSource(
            arxiv_id="2401.00001",
            title="Mock Paper1: Introduction to AI Research",
            authors=["Jack", "Will"],
            abstract_preview="This is a mock abstract for testing purposes in week 1...",
        ),
        PaperSource(
            arxiv_id="2401.00002",
            title="Mock Paper2: Advanced Machine Learning Techniques",
            authors=["Dustin", "Steve"],
            abstract_preview="Another mock abstract demonstrating the API structure...",
        ),
    ]

    return AskResponse(
        answer="This is a mock response for week 1. Real search functionality will be implemented in later phases.",
        sources=mock_sources,
    )
