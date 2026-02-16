import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.config import Settings
from src.dependencies import EmbeddingsDep, GeminiDep, OllamaDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


async def _prepare_chunks_and_sources(request: AskRequest, opensearch_client, embeddings_service):
    query_embedding = None
    search_mode = "bm25"

    if request.use_hybrid:
        try:
            query_embedding = await embeddings_service.embed_query(request.query)
            search_mode = "hybrid"
        except Exception as e:
            logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
            query_embedding = None
            search_mode = "bm25"

    search_results = opensearch_client.search_unified(
        query=request.query,
        query_embedding=query_embedding,
        size=request.top_k,
        from_=0,
        categories=request.categories,
        use_hybrid=request.use_hybrid and query_embedding is not None,
        min_score=0.0,
    )

    chunks = []
    sources_set = set()

    for hit in search_results.get("hits", []):
        arxiv_id = hit.get("arxiv_id", "")
        chunk_data = {
            "arxiv_id": arxiv_id,
            "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
        }
        chunks.append(chunk_data)

        if arxiv_id:
            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
            sources_set.add(pdf_url)

    sources = list(sources_set)
    return chunks, sources, search_mode


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    gemini_client: GeminiDep,
) -> AskResponse:
    settings = Settings()
    provider = request.provider or settings.llm_provider

    try:
        if not opensearch_client.health_check():
            raise HTTPException(status_code=503, detail="Search service is currently unavailable")

        if provider == "ollama":
            try:
                await ollama_client.health_check()
            except Exception as e:
                logger.error(f"Ollama service unavailable: {e}")
                raise HTTPException(status_code=503, detail="LLM service is currently unavailable")

        chunks, sources, search_mode = await _prepare_chunks_and_sources(request, opensearch_client, embeddings_service)

        if not chunks:
            return AskResponse(
                query=request.query,
                answer="I couldn't find any relevant information in the papers to answer your question.",
                sources=[],
                chunks_used=0,
                search_mode=search_mode,
            )

        if provider == "gemini":
            rag_response = await gemini_client.generate_rag_answer(query=request.query, chunks=chunks)
        else:
            model = request.model or settings.ollama_default_model
            rag_response = await ollama_client.generate_rag_answer(query=request.query, chunks=chunks, model=model)

        return AskResponse(
            query=request.query,
            answer=rag_response.get("answer", "Unable to generate answer"),
            sources=rag_response.get("sources", sources),
            chunks_used=len(chunks),
            search_mode=search_mode,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


@router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    gemini_client: GeminiDep,
) -> StreamingResponse:
    settings = Settings()
    provider = request.provider or settings.llm_provider

    async def generate_stream():
        try:
            if not opensearch_client.health_check():
                yield f"data: {json.dumps({'error': 'Search service unavailable'})}\n\n"
                return

            chunks, sources, search_mode = await _prepare_chunks_and_sources(request, opensearch_client, embeddings_service)
            if not chunks:
                yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'done': True})}\n\n"
                return

            yield f"data: {json.dumps({'sources': sources, 'chunks_used': len(chunks), 'search_mode': search_mode})}\n\n"

            if provider == "gemini":
                rag_response = await gemini_client.generate_rag_answer(query=request.query, chunks=chunks)
                answer = rag_response.get("answer", "Unable to generate answer")
                yield f"data: {json.dumps({'chunk': answer})}\n\n"
                yield f"data: {json.dumps({'answer': answer, 'done': True})}\n\n"
                return

            await ollama_client.health_check()
            model = request.model or settings.ollama_default_model
            full_response = ""
            async for chunk in ollama_client.generate_rag_answer_stream(query=request.query, chunks=chunks, model=model):
                if chunk.get("response"):
                    text_chunk = chunk["response"]
                    full_response += text_chunk
                    yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                if chunk.get("done", False):
                    yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                    break

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
