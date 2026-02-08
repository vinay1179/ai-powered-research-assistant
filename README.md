## AI-Powered Research Assistant

Personal implementation based on `production-agentic-rag-course-week1.0` and `production-agentic-rag-course-week2.0`.

### Current Goal
Run Week 1 infrastructure with Week 2 ingestion + PDF parsing on top, and validate the
end-to-end pipeline (API, Airflow, DB, OpenSearch, Ollama).

### Status
- Week 1 + Week 2 merged:
  - arXiv client + factory
  - PDF parser service (Docling) + factory
  - Metadata fetcher pipeline
  - Expanded Paper model + repository queries
  - New schemas (arXiv + PDF parser + API health)
  - `/ask` mock endpoint still available for Week 1
- Airflow ingestion DAGs added (production `arxiv_paper_ingestion`)
- Week 1 + Week 2 notebooks included
- Environment and Docker Compose aligned for local + container networking
  - `.env` setup with custom Postgres + Airflow admin credentials
  - URL-encoded DB passwords for container-safe connection strings
  - Ollama models configured via JSON env

### Airflow Pipeline Diagram
![Airflow ingestion flow](static/week2_data_ingestion_flow.png)

### Testing Performed
- Docker services started successfully via `docker compose up -d`
- Core services healthy: Postgres, OpenSearch, OpenSearch Dashboards, Ollama
- API health verified at `http://localhost:8000/api/v1/health`
- Airflow UI accessible at `http://localhost:8080`
- `arxiv_paper_ingestion` DAG triggered and completed via Airflow UI

### Next Steps
- Run Week 2 notebook (`notebooks/week2/week2_arxiv_integration.ipynb`)
- Execute API and DB smoke tests for stored papers
- Optional: enable OpenSearch indexing (Week 3+)
