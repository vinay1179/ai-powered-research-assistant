import logging
from typing import Dict

from langgraph.runtime import Runtime

from ..prompts import GRADE_DOCUMENTS_PROMPT
from ..state import AgentState
from .utils import build_snippets, call_llm_json

logger = logging.getLogger(__name__)


async def ainvoke_grade_documents_step(state: AgentState, runtime: Runtime) -> Dict:
    query = state.get("current_query") or state.get("query", "")
    documents = list(state.get("documents") or [])
    reasoning_steps = list(state.get("reasoning_steps") or [])
    attempts = int(state.get("retrieval_attempts") or 0)

    if not documents:
        decision = "rewrite_query" if attempts < runtime.context.max_retrieval_attempts else "generate_answer"
        reasoning_steps.append("No documents to grade.")
        return {"routing_decision": decision, "reasoning_steps": reasoning_steps}

    prompt = GRADE_DOCUMENTS_PROMPT.format(query=query, snippets=build_snippets(documents))
    relevant = False
    reason = "No grade response"
    try:
        result = await call_llm_json(runtime, prompt)
        if result:
            relevant = bool(result.get("relevant"))
            reason = result.get("reason", reason)
    except Exception as exc:
        logger.warning("Document grading failed: %s", exc)

    if relevant:
        reasoning_steps.append(f"Documents graded relevant: {reason}")
        return {"routing_decision": "generate_answer", "reasoning_steps": reasoning_steps}

    decision = "rewrite_query" if attempts < runtime.context.max_retrieval_attempts else "generate_answer"
    reasoning_steps.append(f"Documents not relevant: {reason}")
    return {"routing_decision": decision, "reasoning_steps": reasoning_steps}
