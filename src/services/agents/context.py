from dataclasses import dataclass
from typing import Optional

from src.config import Settings
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.gemini.client import GeminiClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient


@dataclass(frozen=True)
class Context:
    """Runtime context for agent dependencies."""

    settings: Settings
    opensearch_client: OpenSearchClient
    embeddings_client: JinaEmbeddingsClient
    ollama_client: OllamaClient
    gemini_client: GeminiClient
    top_k: int
    use_hybrid: bool
    max_retrieval_attempts: int
    langfuse_tracer: Optional[LangfuseTracer] = None
    trace: Optional[object] = None
