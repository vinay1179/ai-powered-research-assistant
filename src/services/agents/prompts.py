GUARDRAIL_PROMPT = """You are a routing assistant for a research paper RAG system.

Decide if the user query is:
- out_of_scope: not related to AI/ML/CS research or inappropriate
- direct_answer: simple question that can be answered without papers
- retrieve: requires retrieving papers

Return ONLY valid JSON: {"decision":"out_of_scope|direct_answer|retrieve","reason":"..."}

Query: {query}
"""

GRADE_DOCUMENTS_PROMPT = """You are grading retrieved research paper snippets for relevance.

Given a query and document snippets, decide if the snippets are relevant enough to answer.
Return ONLY valid JSON: {"relevant": true|false, "reason": "..."}

Query: {query}

Snippets:
{snippets}
"""

REWRITE_QUERY_PROMPT = """Rewrite the user query to improve retrieval of research papers.

Return ONLY valid JSON: {"rewritten_query": "..."}

Original query: {query}
Issue: {reason}
"""

GENERATE_ANSWER_PROMPT = """You are a research assistant.
Answer the question using the provided context. Be concise and cite key points.

Context:
{context}

Question: {query}
"""

DIRECT_ANSWER_PROMPT = """You are a helpful assistant. Answer the question directly.

Question: {query}
"""

OUT_OF_SCOPE_MESSAGE = "This question appears to be outside the scope of the research assistant. Please ask about AI/ML/CS research topics."
