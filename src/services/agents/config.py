from pydantic import BaseModel, Field

from src.config import Settings, get_settings


class GraphConfig(BaseModel):
    """Configuration for the agentic RAG workflow."""

    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60
    temperature: float = 0.0
    top_k: int = 3
    use_hybrid: bool = True
    settings: Settings = Field(default_factory=get_settings)
