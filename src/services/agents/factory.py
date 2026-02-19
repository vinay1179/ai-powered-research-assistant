from src.config import Settings
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.gemini.client import GeminiClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

from .agentic_rag import AgenticRAGService
from .config import GraphConfig


def make_agentic_rag_service(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    ollama_client: OllamaClient,
    gemini_client: GeminiClient,
    langfuse_tracer: LangfuseTracer | None,
    settings: Settings,
) -> AgenticRAGService:
    graph_config = GraphConfig(
        max_retrieval_attempts=settings.agentic_rag.max_retrieval_attempts,
        guardrail_threshold=settings.agentic_rag.guardrail_threshold,
        temperature=settings.agentic_rag.temperature,
        top_k=settings.agentic_rag.top_k,
        use_hybrid=settings.agentic_rag.use_hybrid,
        settings=settings,
    )
    return AgenticRAGService(
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        ollama_client=ollama_client,
        gemini_client=gemini_client,
        langfuse_tracer=langfuse_tracer,
        graph_config=graph_config,
    )
