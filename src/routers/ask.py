from fastapi import APIRouter
from src.config import Settings
from src.dependencies import SessionDep
from src.repositories.paper import PaperRepository
from src.schemas.ask import AskRequest, AskResponse, PaperSource
from src.services.ollama.client import OllamaClient

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, session: SessionDep) -> AskResponse:
    """
    Answer questions using stored paper context when available.
    """
    settings = Settings()
    repo = PaperRepository(session)
    papers = repo.get_processed_papers(limit=3, offset=0)

    sources = [
        PaperSource(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract_preview=(paper.abstract[:240] + "...") if len(paper.abstract) > 240 else paper.abstract,
        )
        for paper in papers
    ]

    context_chunks = []
    for paper in papers:
        if paper.llm_context:
            context_chunks.append(f"Title: {paper.title}\nContext: {paper.llm_context}")
        elif paper.raw_text:
            context_chunks.append(
                f"Title: {paper.title}\nContent: {paper.raw_text[: settings.ollama_context_max_chars]}"
            )
        else:
            context_chunks.append(f"Title: {paper.title}\nAbstract: {paper.abstract}")

    prompt = (
        "You are a research assistant. Use the context below to answer the question.\n\n"
        f"Question: {request.question}\n\n"
        "Context:\n"
        f"{'\n\n'.join(context_chunks)}\n\n"
        "Answer concisely and cite relevant papers if possible."
    )

    ollama = OllamaClient(settings)
    answer_text = (
        "No processed papers are available yet. Please ingest and parse papers before asking questions."
    )
    if context_chunks:
        try:
            response = await ollama.generate(settings.ollama_default_model, prompt)
            if isinstance(response, dict) and response.get("response"):
                answer_text = response["response"]
        except Exception:
            answer_text = "Failed to generate an answer from the available context."

    return AskResponse(answer=answer_text, sources=sources)
