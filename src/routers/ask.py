import json
import logging
import time
from typing import Dict, List, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.config import Settings
from src.dependencies import CacheDep, EmbeddingsDep, GeminiDep, LangfuseDep, OllamaDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.langfuse.tracer import RAGTracer
from src.services.ollama.prompts import RAGPromptBuilder

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


async def _prepare_chunks_and_sources(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
    rag_tracer: RAGTracer,
    trace=None,
) -> Tuple[List[Dict], List[str], str]:
    query_embedding = None
    search_mode = "bm25"

    if request.use_hybrid:
        with rag_tracer.trace_embedding(trace, request.query) as embedding_span:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                search_mode = "hybrid"
            except Exception as exc:
                logger.warning("Failed to generate embeddings, falling back to BM25: %s", exc)
                if embedding_span:
                    rag_tracer.tracer.update_span(embedding_span, output={"success": False, "error": str(exc)})
                query_embedding = None
                search_mode = "bm25"

    with rag_tracer.trace_search(trace, request.query, request.top_k) as search_span:
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
        arxiv_ids = []

        for hit in search_results.get("hits", []):
            arxiv_id = hit.get("arxiv_id", "")
            chunks.append(
                {
                    "arxiv_id": arxiv_id,
                    "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                }
            )

            if arxiv_id:
                arxiv_ids.append(arxiv_id)
                arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

        rag_tracer.end_search(search_span, chunks, arxiv_ids, search_results.get("total", 0))

    sources = list(sources_set)
    return chunks, sources, search_mode


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    gemini_client: GeminiDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> AskResponse:
    settings = Settings()
    provider = request.provider or settings.llm_provider
    effective_model = request.model or settings.ollama_default_model
    if provider == "gemini":
        effective_model = settings.gemini_model

    cache_request = request.model_copy(update={"provider": provider, "model": effective_model})
    rag_tracer = RAGTracer(langfuse_tracer)
    start_time = time.time()

    with rag_tracer.trace_request("api_user", request.query) as trace:
        try:
            if cache_client:
                try:
                    cached_response = await cache_client.find_cached_response(cache_request)
                    if cached_response:
                        rag_tracer.end_request(trace, cached_response.answer, time.time() - start_time)
                        return cached_response
                except Exception as exc:
                    logger.warning("Cache check failed, proceeding with normal flow: %s", exc)

            if not opensearch_client.health_check():
                raise HTTPException(status_code=503, detail="Search service is currently unavailable")

            if provider == "ollama":
                try:
                    await ollama_client.health_check()
                except Exception as exc:
                    logger.error("Ollama service unavailable: %s", exc)
                    raise HTTPException(status_code=503, detail="LLM service is currently unavailable")

            chunks, sources, search_mode = await _prepare_chunks_and_sources(
                request, opensearch_client, embeddings_service, rag_tracer, trace
            )

            if not chunks:
                response = AskResponse(
                    query=request.query,
                    answer="I couldn't find any relevant information in the papers to answer your question.",
                    sources=[],
                    chunks_used=0,
                    search_mode=search_mode,
                )
                rag_tracer.end_request(trace, response.answer, time.time() - start_time)
                return response

            prompt_builder = RAGPromptBuilder()
            with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                if provider == "ollama":
                    try:
                        prompt_data = prompt_builder.create_structured_prompt(request.query, chunks)
                        final_prompt = prompt_data["prompt"]
                    except Exception:
                        final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)
                else:
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)

                rag_tracer.end_prompt(prompt_span, final_prompt)

            with rag_tracer.trace_generation(trace, effective_model, final_prompt) as gen_span:
                if provider == "gemini":
                    rag_response = await gemini_client.generate_rag_answer(query=request.query, chunks=chunks)
                else:
                    rag_response = await ollama_client.generate_rag_answer(
                        query=request.query, chunks=chunks, model=effective_model
                    )

                answer = rag_response.get("answer", "Unable to generate answer")
                rag_tracer.end_generation(gen_span, answer, effective_model)

            response = AskResponse(
                query=request.query,
                answer=answer,
                sources=rag_response.get("sources", sources),
                chunks_used=len(chunks),
                search_mode=search_mode,
            )

            rag_tracer.end_request(trace, answer, time.time() - start_time)

            if cache_client:
                try:
                    await cache_client.store_response(cache_request, response)
                except Exception as exc:
                    logger.warning("Failed to store response in cache: %s", exc)

            return response

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Error in ask endpoint: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process question: {str(exc)}")


@router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    gemini_client: GeminiDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> StreamingResponse:
    settings = Settings()
    provider = request.provider or settings.llm_provider
    effective_model = request.model or settings.ollama_default_model
    if provider == "gemini":
        effective_model = settings.gemini_model

    cache_request = request.model_copy(update={"provider": provider, "model": effective_model})

    async def generate_stream():
        rag_tracer = RAGTracer(langfuse_tracer)
        start_time = time.time()

        with rag_tracer.trace_request("api_user", request.query) as trace:
            try:
                if cache_client:
                    try:
                        cached_response = await cache_client.find_cached_response(cache_request)
                        if cached_response:
                            metadata_response = {
                                "sources": cached_response.sources,
                                "chunks_used": cached_response.chunks_used,
                                "search_mode": cached_response.search_mode,
                            }
                            yield f"data: {json.dumps(metadata_response)}\n\n"

                            for chunk in cached_response.answer.split():
                                yield f"data: {json.dumps({'chunk': chunk + ' '})}\n\n"

                            yield f"data: {json.dumps({'answer': cached_response.answer, 'done': True})}\n\n"
                            rag_tracer.end_request(trace, cached_response.answer, time.time() - start_time)
                            return
                    except Exception as exc:
                        logger.warning("Cache check failed, proceeding with normal flow: %s", exc)

                if not opensearch_client.health_check():
                    yield f"data: {json.dumps({'error': 'Search service unavailable'})}\n\n"
                    return

                chunks, sources, search_mode = await _prepare_chunks_and_sources(
                    request, opensearch_client, embeddings_service, rag_tracer, trace
                )
                if not chunks:
                    yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'done': True})}\n\n"
                    return

                yield f"data: {json.dumps({'sources': sources, 'chunks_used': len(chunks), 'search_mode': search_mode})}\n\n"

                prompt_builder = RAGPromptBuilder()
                with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                    if provider == "ollama":
                        final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)
                    else:
                        final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)
                    rag_tracer.end_prompt(prompt_span, final_prompt)

                with rag_tracer.trace_generation(trace, effective_model, final_prompt) as gen_span:
                    if provider == "gemini":
                        rag_response = await gemini_client.generate_rag_answer(query=request.query, chunks=chunks)
                        answer = rag_response.get("answer", "Unable to generate answer")
                        rag_tracer.end_generation(gen_span, answer, effective_model)
                        yield f"data: {json.dumps({'chunk': answer})}\n\n"
                        yield f"data: {json.dumps({'answer': answer, 'done': True})}\n\n"
                        rag_tracer.end_request(trace, answer, time.time() - start_time)
                    else:
                        await ollama_client.health_check()
                        full_response = ""
                        async for chunk in ollama_client.generate_rag_answer_stream(
                            query=request.query, chunks=chunks, model=effective_model
                        ):
                            if chunk.get("response"):
                                text_chunk = chunk["response"]
                                full_response += text_chunk
                                yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                            if chunk.get("done", False):
                                rag_tracer.end_generation(gen_span, full_response, effective_model)
                                yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                                rag_tracer.end_request(trace, full_response, time.time() - start_time)
                                break

                if cache_client:
                    try:
                        response_to_cache = AskResponse(
                            query=request.query,
                            answer=full_response if provider == "ollama" else answer,
                            sources=sources,
                            chunks_used=len(chunks),
                            search_mode=search_mode,
                        )
                        await cache_client.store_response(cache_request, response_to_cache)
                    except Exception as exc:
                        logger.warning("Failed to store streaming response in cache: %s", exc)

            except Exception as exc:
                logger.error("Streaming error: %s", exc, exc_info=True)
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
