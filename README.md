# AI-Powered Research Assistant

## ✅ Implementations Completed

- **Week 1 Infrastructure**
  - Docker Compose stack: FastAPI, Postgres, OpenSearch, Airflow, Ollama
  - Health checks and local service orchestration
- **Week 2 Ingestion + PDF Parsing**
  - arXiv client + factory
  - Docling-based PDF parser service
  - Metadata fetcher pipeline
  - Postgres storage with extended paper fields
  - LLM context fields stored in DB
- **Week 3 Search Foundation**
  - OpenSearch service layer (client, index config, query builder)
  - Search API: `POST /api/v1/search` with BM25 + filters + highlights + pagination
  - Index creation + OpenSearch health checks on startup
  - Airflow indexing task
  - 6 PM UTC retry DAG for failed PDFs (reprocess, rebuild LLM context, reindex)
- **Week 4 Hybrid Search**
  - Section-based chunking + chunk indexing service
  - Jina embeddings client integration for vector search
  - Hybrid OpenSearch index with RRF pipeline
  - Hybrid search API: `POST /api/v1/hybrid-search`
  - Airflow hybrid indexing task (chunks + embeddings)
- **Week 5 RAG + Multi-LLM**
  - RAG endpoints: `POST /api/v1/ask` and `POST /api/v1/stream`
  - Provider selection per request (`ollama` or `gemini`)
  - RAG prompt builder + structured response parsing
  - Gradio UI for interactive RAG
- **Week 6 Observability + Cache**
  - Langfuse tracing integrated into the RAG pipeline
  - Redis exact-match caching for `/ask` and `/stream`
  - Langfuse + ClickHouse services added for monitoring
- **Week 7 LangGraph + Guardrails**
  - Agentic workflow orchestration with LangGraph for multi-step retrieval
  - Guardrail checks to prevent low-quality or unsafe responses
  - Agentic ask endpoint: `POST /api/v1/agentic-ask`
- **Local PDF Testing**
  - Local-only ingestion endpoint for PDF testing
  - Local PDF directory mount for container parsing

---

## ✅ Tests Completed

- LLM context generation via Gemini (summary/key points/context persisted)
- Airflow arXiv ingestion (5 PDFs, Gemini enabled; 1 skipped due to >30 pages)
- Manual arXiv ingestion (5 PDFs, Gemini enabled; 1 skipped due to >45 pages)
- OpenSearch indexing verified for arXiv papers
- Hybrid chunk indexing verified in OpenSearch (`arxiv-papers-chunks`)
- Hybrid chunk indexing via Jina embeddings (rate-limit errors observed, partial success)
- Hybrid search endpoint validated (`POST /api/v1/hybrid-search`)
- RAG ask endpoint validated with Gemini (`POST /api/v1/ask`)
- OpenSearch Dashboards UI verified at `http://localhost:5601`
- Gradio UI verified at `http://localhost:7861`

---

## 🔭 Future Enhancements

- Enforce strict JSON responses from LLMs to avoid markdown fallbacks
- Add a smaller, dedicated LLM input cap (separate from PDF max chars)
- Implement chunking + map-reduce summarization for long PDFs
- Add retries/backoff + partial results when LLM times out
- Cache LLM outputs to avoid recompute during re-ingestion
- Allow per-model timeout and context limits via env settings
- Add a lightweight/faster model option for quick summaries
- Evaluate storing vector embeddings in Postgres (pgvector) for backup/analytics use
- Research other embeddings for better performance vs Jina
- Explore alternatives to Docling for faster PDF parsing
- Explore alternatives to BM25 and clustering algorithms for better grouping
