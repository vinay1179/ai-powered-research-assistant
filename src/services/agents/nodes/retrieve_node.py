import logging
from typing import Dict

from langgraph.runtime import Runtime

from ..state import AgentState

logger = logging.getLogger(__name__)


async def ainvoke_retrieve_step(state: AgentState, runtime: Runtime) -> Dict:
    query = state.get("current_query") or state.get("query", "")
    reasoning_steps = list(state.get("reasoning_steps") or [])
    attempts = int(state.get("retrieval_attempts") or 0) + 1

    try:
        query_embedding = await runtime.context.embeddings_client.embed_query(query)
        search_results = runtime.context.opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=runtime.context.top_k,
            use_hybrid=runtime.context.use_hybrid,
        )
        documents = search_results.get("hits", [])
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        documents = []

    sources = []
    for doc in documents:
        arxiv_id = doc.get("arxiv_id", "")
        if arxiv_id:
            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            sources.append(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

    reasoning_steps.append(f"Retrieved {len(documents)} documents (attempt {attempts}).")

    return {
        "documents": documents,
        "sources": list(dict.fromkeys(sources)),
        "retrieval_attempts": attempts,
        "reasoning_steps": reasoning_steps,
    }
