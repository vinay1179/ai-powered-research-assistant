import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from langfuse import Langfuse
from src.config import Settings

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Wrapper for Langfuse tracing client."""

    def __init__(self, settings: Settings):
        self.settings = settings.langfuse
        self.client: Optional[Langfuse] = None

        if self.settings.enabled and self.settings.public_key and self.settings.secret_key:
            try:
                self.client = Langfuse(
                    public_key=self.settings.public_key,
                    secret_key=self.settings.secret_key,
                    host=self.settings.host,
                    flush_at=self.settings.flush_at,
                    flush_interval=self.settings.flush_interval,
                    debug=self.settings.debug,
                )
                logger.info("Langfuse tracing initialized (host: %s)", self.settings.host)
            except Exception as exc:
                logger.error("Failed to initialize Langfuse: %s", exc)
                self.client = None
        else:
            logger.info("Langfuse tracing disabled or missing credentials")

    @contextmanager
    def trace_rag_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing a RAG request."""
        if not self.client:
            yield None
            return

        try:
            trace = self.client.trace(
                name="rag_request",
                input={"query": query},
                metadata=metadata or {},
                user_id=user_id,
                session_id=session_id,
            )
            yield trace
        except Exception as exc:
            logger.error("Error creating Langfuse trace: %s", exc)
            yield None

    def create_span(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create a span within a trace."""
        if not trace or not self.client:
            return None

        try:
            return self.client.span(
                trace_id=trace.trace_id,
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
        except Exception as exc:
            logger.error("Error creating span %s: %s", name, exc)
            return None

    def create_generation(
        self,
        trace,
        name: str,
        model: str,
        input_data: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None,
    ):
        """Create a generation (LLM call) within a trace."""
        if not trace or not self.client:
            return None

        try:
            return self.client.generation(
                trace_id=trace.trace_id,
                name=name,
                model=model,
                input=input_data,
                output=output,
                metadata=metadata or {},
                usage=usage,
            )
        except Exception as exc:
            logger.error("Error creating generation %s: %s", name, exc)
            return None

    def update_span(
        self,
        span,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ):
        """Update a span with output or additional metadata."""
        if not span:
            return

        try:
            if output is not None:
                span.update(output=output)
            if metadata:
                span.update(metadata=metadata)
            if level:
                span.update(level=level)
            if status_message:
                span.update(status_message=status_message)
        except Exception as exc:
            logger.error("Error updating span: %s", exc)

    def flush(self):
        """Flush any pending traces."""
        if self.client:
            try:
                self.client.flush()
            except Exception as exc:
                logger.error("Error flushing Langfuse: %s", exc)

    def shutdown(self):
        """Shutdown the Langfuse client."""
        if self.client:
            try:
                self.client.flush()
                self.client.shutdown()
            except Exception as exc:
                logger.error("Error shutting down Langfuse: %s", exc)
