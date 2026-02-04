## AI-Powered Research Assistant

Personal implementation based on `production-agentic-rag-course-week1.0`.

### Week 1 Goal
Set up the full local infrastructure stack (API, Postgres, OpenSearch, Airflow, Ollama) and verify service health.

### Status
- Core scaffolding added:
  - Config, DB interfaces, Postgres adapter, DB factory, and dependencies
  - Paper model, schemas, and repository
  - Health/ping + ask + papers routers
  - Ollama client service
  - Exceptions and middleware utilities
- Infra + tooling added:
  - Dockerfile, Compose stack, Airflow Dockerfile/entrypoint
  - Airflow DAG + init SQL + requirements
  - Week 1 notebook + README
  - Makefile, .env.example, .gitignore
- Git initialized and pushed to `origin/main`

### Next Steps
- Add `.env` with custom DB credentials
- Run `docker compose up -d` and verify health checks
