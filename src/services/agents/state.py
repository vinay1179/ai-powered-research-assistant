from typing import List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    current_query: str
    retrieval_attempts: int
    documents: List[dict]
    sources: List[str]
    reasoning_steps: List[str]
    answer: Optional[str]
    routing_decision: Optional[str]
    direct_answer: bool
