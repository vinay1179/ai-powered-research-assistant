import logging
from typing import Dict

from langgraph.runtime import Runtime

from ..prompts import REWRITE_QUERY_PROMPT
from ..state import AgentState
from .utils import call_llm_json

logger = logging.getLogger(__name__)


async def ainvoke_rewrite_query_step(state: AgentState, runtime: Runtime) -> Dict:
    query = state.get("current_query") or state.get("query", "")
    reasoning_steps = list(state.get("reasoning_steps") or [])

    rewritten = query
    reason = "No rewrite response"
    try:
        result = await call_llm_json(runtime, REWRITE_QUERY_PROMPT.format(query=query, reason="Improve retrieval relevance"))
        if result and result.get("rewritten_query"):
            rewritten = result["rewritten_query"]
            reason = "Rewrote query for better retrieval"
    except Exception as exc:
        logger.warning("Query rewrite failed: %s", exc)

    reasoning_steps.append(f"{reason}: {rewritten}")
    return {"current_query": rewritten, "reasoning_steps": reasoning_steps}
