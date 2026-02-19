import logging
from typing import Dict

from langgraph.runtime import Runtime

from ..prompts import GUARDRAIL_PROMPT
from ..state import AgentState
from .utils import call_llm_json

logger = logging.getLogger(__name__)


async def ainvoke_guardrail_step(state: AgentState, runtime: Runtime) -> Dict:
    query = state.get("current_query") or state.get("query", "")
    reasoning_steps = list(state.get("reasoning_steps") or [])

    decision = "retrieve"
    reason = "Defaulted to retrieval"
    try:
        result = await call_llm_json(runtime, GUARDRAIL_PROMPT.format(query=query))
        if result:
            decision = result.get("decision", decision)
            reason = result.get("reason", reason)
    except Exception as exc:
        logger.warning("Guardrail decision failed: %s", exc)

    reasoning_steps.append(f"Guardrail decision: {decision} ({reason})")

    return {
        "routing_decision": decision,
        "reasoning_steps": reasoning_steps,
        "direct_answer": decision == "direct_answer",
    }


def continue_after_guardrail(state: AgentState) -> str:
    return state.get("routing_decision") or "retrieve"
