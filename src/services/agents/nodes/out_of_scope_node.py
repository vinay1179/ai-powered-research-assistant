from typing import Dict

from langgraph.runtime import Runtime

from ..prompts import OUT_OF_SCOPE_MESSAGE
from ..state import AgentState


async def ainvoke_out_of_scope_step(state: AgentState, runtime: Runtime) -> Dict:
    reasoning_steps = list(state.get("reasoning_steps") or [])
    reasoning_steps.append("Marked as out of scope.")
    return {
        "answer": OUT_OF_SCOPE_MESSAGE,
        "reasoning_steps": reasoning_steps,
        "sources": [],
    }
