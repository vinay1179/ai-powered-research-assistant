import logging
import time
from typing import Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.gemini.client import GeminiClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

from .config import GraphConfig
from .context import Context
from .nodes import (
    ainvoke_generate_answer_step,
    ainvoke_grade_documents_step,
    ainvoke_guardrail_step,
    ainvoke_out_of_scope_step,
    ainvoke_retrieve_step,
    ainvoke_rewrite_query_step,
    continue_after_guardrail,
)
from .state import AgentState

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """Agentic RAG service using LangGraph."""

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        embeddings_client: JinaEmbeddingsClient,
        ollama_client: OllamaClient,
        gemini_client: GeminiClient,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        graph_config: Optional[GraphConfig] = None,
    ):
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.ollama = ollama_client
        self.gemini = gemini_client
        self.langfuse_tracer = langfuse_tracer
        self.graph_config = graph_config or GraphConfig()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState, context_schema=Context)

        workflow.add_node("guardrail", ainvoke_guardrail_step)
        workflow.add_node("out_of_scope", ainvoke_out_of_scope_step)
        workflow.add_node("retrieve", ainvoke_retrieve_step)
        workflow.add_node("grade_documents", ainvoke_grade_documents_step)
        workflow.add_node("rewrite_query", ainvoke_rewrite_query_step)
        workflow.add_node("generate_answer", ainvoke_generate_answer_step)

        workflow.add_edge(START, "guardrail")
        workflow.add_conditional_edges(
            "guardrail",
            continue_after_guardrail,
            {
                "continue": "retrieve",
                "retrieve": "retrieve",
                "direct_answer": "generate_answer",
                "out_of_scope": "out_of_scope",
            },
        )
        workflow.add_edge("out_of_scope", END)
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            lambda state: state.get("routing_decision", "generate_answer"),
            {"generate_answer": "generate_answer", "rewrite_query": "rewrite_query"},
        )
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    async def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_hybrid: Optional[bool] = None,
        user_id: str = "api_user",
    ) -> Dict[str, object]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        start_time = time.time()
        trace_id = None

        state_input: AgentState = {
            "query": query,
            "current_query": query,
            "retrieval_attempts": 0,
            "documents": [],
            "sources": [],
            "reasoning_steps": [],
            "answer": None,
            "routing_decision": None,
            "direct_answer": False,
        }

        resolved_top_k = top_k if top_k is not None else self.graph_config.top_k
        resolved_use_hybrid = use_hybrid if use_hybrid is not None else self.graph_config.use_hybrid

        runtime_context = Context(
            settings=self.graph_config.settings,
            opensearch_client=self.opensearch,
            embeddings_client=self.embeddings,
            ollama_client=self.ollama,
            gemini_client=self.gemini,
            top_k=resolved_top_k,
            use_hybrid=resolved_use_hybrid,
            max_retrieval_attempts=self.graph_config.max_retrieval_attempts,
            langfuse_tracer=self.langfuse_tracer,
        )

        with self._trace_request(query=query, user_id=user_id) as trace:
            if trace:
                trace_id = self.langfuse_tracer.get_trace_id(trace)

            result = await self.graph.ainvoke(state_input, context=runtime_context)

            if trace:
                duration = time.time() - start_time
                try:
                    trace.update(
                        output={
                            "answer": result.get("answer"),
                            "sources_count": len(result.get("sources") or []),
                            "retrieval_attempts": result.get("retrieval_attempts", 0),
                            "reasoning_steps": result.get("reasoning_steps", []),
                            "execution_time": duration,
                        }
                    )
                except Exception:
                    pass

        return {
            "query": query,
            "answer": result.get("answer") or "",
            "sources": result.get("sources") or [],
            "reasoning_steps": result.get("reasoning_steps") or [],
            "retrieval_attempts": result.get("retrieval_attempts", 0),
            "trace_id": trace_id,
        }

    def _trace_request(self, query: str, user_id: str):
        if not self.langfuse_tracer:
            return _NullContext()
        return self.langfuse_tracer.trace_rag_request(
            query=query,
            user_id=user_id,
            session_id=f"agentic_{user_id}",
            metadata={"agentic": True},
        )


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
