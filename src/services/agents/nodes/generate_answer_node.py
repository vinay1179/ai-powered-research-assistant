import logging
from typing import Dict

from langgraph.runtime import Runtime

from ..prompts import DIRECT_ANSWER_PROMPT, GENERATE_ANSWER_PROMPT
from ..state import AgentState
from .utils import build_context, call_llm_text

logger = logging.getLogger(__name__)


async def ainvoke_generate_answer_step(state: AgentState, runtime: Runtime) -> Dict:
    query = state.get("current_query") or state.get("query", "")
    documents = list(state.get("documents") or [])
    reasoning_steps = list(state.get("reasoning_steps") or [])
    direct_answer = bool(state.get("direct_answer"))

    if direct_answer:
        prompt = DIRECT_ANSWER_PROMPT.format(query=query)
        reasoning_steps.append("Generated direct answer without retrieval.")
    else:
        context = build_context(documents, max_chars=runtime.context.settings.ollama_context_max_chars)
        prompt = GENERATE_ANSWER_PROMPT.format(context=context or "No context available.", query=query)
        reasoning_steps.append("Generated answer using retrieved documents.")

    answer = ""
    try:
        answer = await call_llm_text(runtime, prompt)
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        answer = "I encountered an error while generating the answer. Please try again."

    return {"answer": answer or "No answer generated.", "reasoning_steps": reasoning_steps}
