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
  - **6 PM UTC retry DAG** that reprocesses failed PDFs, rebuilds LLM context, and reindexes
- **Week 4 Hybrid Search**
  - Section-based chunking + chunk indexing service
  - Jina embeddings client integration for vector search
  - Hybrid OpenSearch index with RRF pipeline
  - Hybrid search API: `POST /api/v1/hybrid-search`
  - Airflow hybrid indexing task (chunks + embeddings)
- **Local PDF Testing + Gemini LLM**
  - Local-only ingestion endpoint for PDF testing
  - Gemini provider integration (keeps Ollama available)
  - Local PDF directory mount for container parsing

---

## ✅ Tests Completed

- Local PDF ingestion (2 PDFs) with parsing + DB storage
- LLM context generation via Gemini (summary, key points, context persisted)
- OpenSearch indexing validated (raw_text stored)
- BM25 scoring verified via OpenSearch `_explain`
- OpenSearch Dashboards UI verified at `http://localhost:5601`

---

## 🔭 Future Enhancements (LLM Context Building)

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
# AI-Powered Research Assistant

---

## ✅ Tests Completed

- Local PDF ingestion (2 PDFs) with parsing + DB storage
- LLM context generation via Gemini (summary, key points, context persisted)
- OpenSearch indexing validated (raw_text stored)
- BM25 scoring verified via OpenSearch `_explain`
- OpenSearch Dashboards UI verified at `http://localhost:5601`

---

## 🔭 Future Enhancements (LLM Context Building)

- Enforce strict JSON responses from LLMs to avoid markdown fallbacks
- Add a smaller, dedicated LLM input cap (separate from PDF max chars)
- Implement chunking + map-reduce summarization for long PDFs
- Add retries/backoff + partial results when LLM times out
- Cache LLM outputs to avoid recompute during re-ingestion
- Allow per-model timeout and context limits via env settings
- Add a lightweight/faster model option for quick summaries
- Explore alternatives to Docling for faster PDF parsing
- Explore alternatives to BM25 and clustering algorithms for better grouping
# AI-Powered Research Assistant

Personal implementation built on the production-agentic-rag-course Week 1–3 releases.

---

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
  - **6 PM UTC retry DAG** that reprocesses failed PDFs, rebuilds LLM context, and reindexes
- **Local PDF Testing + Gemini LLM**
  - Local-only ingestion endpoint for PDF testing
  - Gemini provider integration (keeps Ollama available)
  - Local PDF directory mount for container parsing

---

## 🔭 Future Enhancements (LLM Context Building)

- Enforce strict JSON responses from Ollama to avoid markdown fallbacks
- Add a smaller, dedicated LLM input cap (separate from PDF max chars)
- Implement chunking + map-reduce summarization for long PDFs
- Add retries/backoff + partial results when LLM times out
- Cache LLM outputs to avoid recompute during re-ingestion
- Allow per-model timeout and context limits via env settings
- Add a lightweight/faster model option for quick summaries
- Explore alternatives to Docling for faster PDF parsing
- Explore alternatives to BM25 and clustering algorithms for better grouping

---

## ✅ Tests Completed

- Local PDF ingestion (2 PDFs) with parsing + DB storage
- LLM context generation via Gemini (summary, key points, context persisted)
- OpenSearch indexing validated (raw_text stored)
- BM25 scoring verified via OpenSearch `_explain`
- OpenSearch Dashboards UI verified at `http://localhost:5601`

---

## Quick Start

```bash
cp .env.example .env
uv sync
docker compose up --build -d
```

---

## Useful Links

- API docs: http://localhost:8000/docs
- Airflow: http://localhost:8080
- OpenSearch Dashboards: http://localhost:5601
# The Mother of AI Project
## Phase 1 RAG Systems: arXiv Paper Curator

<div align="center">
  <h3>A Learner-Focused Journey into Production RAG Systems</h3>
  <p>Learn to build modern AI systems from the ground up through hands-on implementation</p>
  <p>Master the most in-demand AI engineering skills: <strong>RAG (Retrieval-Augmented Generation)</strong></p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenSearch-2.19-orange.svg" alt="OpenSearch">
  <img src="https://img.shields.io/badge/Docker-Compose-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/Status-Week%203%20Keyword%20Search-brightgreen.svg" alt="Status">
</p>

</br>

<p align="center">
  <a href="#-about-this-course">
    <img src="static/mother_of_ai_project_rag_architecture.gif" alt="RAG Architecture" width="700">
  </a>
</p>

## 📖 About This Course

This is a **learner-focused project** where you'll build a complete research assistant system that automatically fetches academic papers, understands their content, and answers your research questions using advanced RAG techniques.

**The arXiv Paper Curator** will teach you to build a **production-grade RAG system using industry best practices**. Unlike tutorials that jump straight to vector search, we follow the **professional path**: master keyword search foundations first, then enhance with vectors for hybrid retrieval.

> **🎯 The Professional Difference:** We build RAG systems the way successful companies do - solid search foundations enhanced with AI, not AI-first approaches that ignore search fundamentals.

By the end of this course, you'll have your own AI research assistant and the deep technical skills to build production RAG systems for any domain.

---

## ✅ Implementations In This Repo

This project integrates Week 1–3 features on top of the base code:

- **Week 1 Infrastructure**: Docker Compose for FastAPI, Postgres, OpenSearch, Airflow, Ollama.
- **Week 2 Ingestion**: arXiv client, Docling PDF parsing, metadata fetcher, Postgres storage, LLM context fields.
- **Week 3 Search Foundation**:
  - OpenSearch service layer (client, index config, query builder).
  - `POST /api/v1/search` with BM25, filters, highlights, pagination.
  - Index creation on startup + health checks.
  - Airflow indexing task and OpenSearch metrics in reports.
  - **6 PM UTC retry DAG** that reprocesses failed PDFs, rebuilds LLM context, and reindexes.

If you want a detailed validation checklist or test commands, see `notebooks/week3/README.md`.

---

## 🚀 Quick Start

### **📋 Prerequisites**
- **Docker Desktop** (with Docker Compose)  
- **Python 3.12+**
- **UV Package Manager** ([Install Guide](https://docs.astral.sh/uv/getting-started/installation/))
- **8GB+ RAM** and **20GB+ free disk space**

### **⚡ Get Started**
```bash
# 1. Clone and setup
git clone <repository-url>
cd arxiv-paper-curator

# 2. Configure environment (IMPORTANT!)
cp .env.example .env
# The .env file contains all necessary configuration for OpenSearch, 
# arXiv API, and service connections. Defaults work out of the box.

# 3. Install dependencies
uv sync

# 4. Start all services
docker compose up --build -d

# 5. Verify everything works
curl http://localhost:8000/health
```


**📥 Clone a specific week's release:**
```bash
# Clone a specific week's code
git clone --branch <WEEK_TAG> https://github.com/jamwithai/arxiv-paper-curator
cd arxiv-paper-curator
uv sync
docker compose down -v
docker compose up --build -d

# Replace <WEEK_TAG> with: week1.0, week2.0, etc.
```

### **📊 Access Your Services**

| Service | URL | Purpose |
|---------|-----|---------|
| **API Documentation** | http://localhost:8000/docs | Interactive API testing |
| **Airflow Dashboard** | http://localhost:8080 | Workflow management |
| **OpenSearch Dashboards** | http://localhost:5601 | Hybrid search engine UI |

#### **NOTE**: Check airflow/simple_auth_manager_passwords.json.generated for Airflow username and password
---

## 📚 Week 1: Infrastructure Foundation ✅

**Start here!** Master the infrastructure that powers modern RAG systems.

### **🎯 Learning Objectives**
- Complete infrastructure setup with Docker Compose
- FastAPI development with automatic documentation and health checks
- PostgreSQL database configuration and management
- OpenSearch hybrid search engine setup
- Ollama local LLM service configuration
- Service orchestration and health monitoring
- Professional development environment with code quality tools

### **🏗️ Architecture Overview**

<p align="center">
  <img src="static/week1_infra_setup.png" alt="Week 1 Infrastructure Setup" width="800">
</p>

**Infrastructure Components:**
- **FastAPI**: REST endpoints with async support (Port 8000)  
- **PostgreSQL 16**: Paper metadata storage (Port 5432)
- **OpenSearch 2.19**: Search engine with dashboards (Ports 9200, 5601)
- **Apache Airflow 3.0**: Workflow orchestration (Port 8080)
- **Ollama**: Local LLM server (Port 11434)

### **📓 Setup Guide**

```bash
# Launch the Week 1 notebook
uv run jupyter notebook notebooks/week1/week1_setup.ipynb
```

### **✅ Success Criteria**
Complete when you can:
- [ ] Start all services with `docker compose up -d`
- [ ] Access API docs at http://localhost:8000/docs  
- [ ] Login to Airflow at http://localhost:8080
- [ ] Browse OpenSearch at http://localhost:5601
- [ ] All tests pass: `uv run pytest`

### **📖 Deep Dive**
**Blog Post:** [The Infrastructure That Powers RAG Systems](https://jamwithai.substack.com/p/the-infrastructure-that-powers-rag) - Detailed walkthrough and production insights

---

## 📚 Week 2: Data Ingestion Pipeline ✅

**Building on Week 1 infrastructure:** Learn to fetch, process, and store academic papers automatically.

### **🎯 Learning Objectives**
- arXiv API integration with rate limiting and retry logic
- Scientific PDF parsing using Docling
- Automated data ingestion pipelines with Apache Airflow
- Metadata extraction and storage workflows
- Complete paper processing from API to database

### **🏗️ Architecture Overview**

<p align="center">
  <img src="static/week2_data_ingestion_flow.png" alt="Week 2 Data Ingestion Architecture" width="800">
</p>

**Data Pipeline Components:**
- **MetadataFetcher**: 🎯 Main orchestrator coordinating the entire pipeline
- **ArxivClient**: Rate-limited paper fetching with retry logic
- **PDFParserService**: Docling-powered scientific document processing  
- **Airflow DAGs**: Automated daily paper ingestion workflows
- **PostgreSQL Storage**: Structured paper metadata and content

### **📓 Implementation Guide**

```bash
# Launch the Week 2 notebook  
uv run jupyter notebook notebooks/week2/week2_arxiv_integration.ipynb
```

### **💻 Code Examples**

**arXiv API Integration:**
```python
# Example: Fetch papers with rate limiting
from src.services.arxiv.factory import make_arxiv_client

async def fetch_recent_papers():
    client = make_arxiv_client()
    papers = await client.search_papers(
        query="cat:cs.AI",
        max_results=10,
        from_date="20240801",
        to_date="20240807"
    )
    return papers
```

**PDF Processing Pipeline:**
```python
# Example: Parse PDF with Docling
from src.services.pdf_parser.factory import make_pdf_parser_service

async def process_paper_pdf(pdf_url: str):
    parser = make_pdf_parser_service()
    parsed_content = await parser.parse_pdf_from_url(pdf_url)
    return parsed_content  # Structured content with text, tables, figures
```

**Complete Ingestion Workflow:**
```python
# Example: Full paper ingestion pipeline
from src.services.metadata_fetcher import make_metadata_fetcher

async def ingest_papers():
    fetcher = make_metadata_fetcher()
    results = await fetcher.fetch_and_store_papers(
        query="cat:cs.AI",
        max_results=5,
        from_date="20240807"
    )
    return results  # Papers stored in database with full content
```

### **✅ Success Criteria**
Complete when you can:
- [ ] Fetch papers from arXiv API: Test in Week 2 notebook
- [ ] Parse PDF content with Docling: View extracted structured content
- [ ] Run Airflow DAG: `arxiv_paper_ingestion` executes successfully
- [ ] Verify database storage: Papers appear in PostgreSQL with full content
- [ ] API endpoints work: `/papers` returns stored papers with metadata

### **📖 Deep Dive**
**Blog Post:** [Building Data Ingestion Pipelines for RAG](https://jamwithai.substack.com/p/bringing-your-rag-system-to-life) - arXiv API integration and PDF processing

---

## 📚 Week 3: Keyword Search First - The Critical Foundation ⚡

> **🚨 The 90% Problem:** Most RAG systems jump straight to vector search and miss the foundation that powers the best retrieval systems. We're doing it right!

**Building on Weeks 1-2 foundation:** Implement the keyword search foundation that professional RAG systems rely on.

### **🎯 Why Keyword Search First?**

**The Reality Check:** Vector search alone is not enough. The most effective RAG systems use **hybrid retrieval** - combining keyword search (BM25) with vector search. Here's why we start with keywords:

1. **🔍 Exact Match Power:** Keywords excel at finding specific terms, technical jargon, and precise phrases
2. **📊 Interpretable Results:** You can understand exactly why a document was retrieved  
3. **⚡ Speed & Efficiency:** BM25 is computationally fast and doesn't require expensive embedding models
4. **🎯 Domain Knowledge:** Technical papers often require exact terminology matches that vector search might miss
5. **📈 Production Reality:** Companies like Elasticsearch, Algolia, and enterprise search all use keyword search as their foundation

### **🏗️ Week 3 Architecture Overview**

<p align="center">
  <img src="static/week3_opensearch_flow.png" alt="Week 3 OpenSearch Flow Architecture" width="800">
  <br>
  <em>Complete Week 3 architecture showing the OpenSearch integration flow</em>
</p>

**Search Infrastructure:** Master full-text search with OpenSearch before adding vector complexity.

#### **🎯 Learning Objectives**
- **Foundation First:** Why keyword search is essential for RAG systems
- **OpenSearch Mastery:** Index management, mappings, and search optimization
- **BM25 Algorithm:** Understanding the math behind effective keyword search
- **Query DSL:** Building complex search queries with filters and boosting
- **Search Analytics:** Measuring search relevance and performance
- **Production Patterns:** How real companies structure their search systems

#### **Key Components**
- `src/services/opensearch/`: Professional search service implementation
- `src/routers/search.py`: Search API endpoints with BM25 scoring
- `notebooks/week3/`: Complete OpenSearch integration guide  
- **Search Quality Metrics:** Precision, recall, and relevance scoring

#### **💡 The Pedagogical Approach**
```
Week 3: Master keyword search (BM25) ← YOU ARE HERE
Week 4: Add intelligent chunking strategies  
Week 5: Introduce vector embeddings for hybrid retrieval
Week 6: Optimize the complete hybrid system
```

**This progression mirrors how successful companies build search systems - solid foundation first, then enhance with advanced techniques.**

### **📓 Week 3 Implementation Guide**

```bash
# Launch the Week 3 notebook
uv run jupyter notebook notebooks/week3/week3_opensearch.ipynb
```

### **💻 Code Examples**

**BM25 Search Implementation:**
```python
# Example: Search papers with BM25 scoring
from src.services.opensearch.factory import make_opensearch_client

async def search_papers():
    client = make_opensearch_client()
    results = await client.search_papers(
        query="transformer attention mechanism",
        max_results=10,
        categories=["cs.AI", "cs.LG"]
    )
    return results  # Papers ranked by BM25 relevance
```

**Search API Usage:**
```python
# Example: Use the search endpoint
import httpx

async def query_papers():
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/api/v1/search", json={
            "query": "neural networks optimization",
            "max_results": 5,
            "latest_papers": True
        })
        return response.json()
```

### **✅ Success Criteria**
Complete when you can:
- [ ] Index papers in OpenSearch: Papers searchable via OpenSearch Dashboards
- [ ] Search via API: `/search` endpoint returns relevant papers with BM25 scoring
- [ ] Filter by categories: Search within specific arXiv categories (cs.AI, cs.LG, etc.)
- [ ] Sort by relevance or date: Toggle between BM25 scoring and latest papers
- [ ] View search analytics: Understanding why papers matched your query

### **Future Weeks Overview** (Weeks 4-6)
- **Week 4:** Chunking strategies and hybrid retrieval (combining keyword + vector search)
- **Week 5:** Full RAG pipeline with LLM integration and prompt optimization
- **Week 6:** Observability with Langfuse and evaluation systems

---

## ⚙️ Configuration Management

### **Environment Configuration**

The project uses a **unified `.env` file** with nested configuration structure to manage settings across all services.

#### **Configuration Structure**
```bash
# Application Settings
DEBUG=true
ENVIRONMENT=development

# arXiv API (Week 2)
ARXIV__MAX_RESULTS=15
ARXIV__SEARCH_CATEGORY=cs.AI
ARXIV__RATE_LIMIT_DELAY=3.0

# PDF Parser (Week 2)  
PDF_PARSER__MAX_PAGES=30
PDF_PARSER__DO_OCR=false

# OpenSearch (Week 3)
OPENSEARCH__HOST=http://opensearch:9200
OPENSEARCH__INDEX_NAME=arxiv-papers

# Services
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=llama3.2:1b
```

#### **Key Configuration Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `true` | Debug mode for development |
| `ARXIV__MAX_RESULTS` | `15` | Papers to fetch per API call |
| `ARXIV__SEARCH_CATEGORY` | `cs.AI` | arXiv category to search |
| `PDF_PARSER__MAX_PAGES` | `30` | Max pages to process per PDF |
| `OPENSEARCH__INDEX_NAME` | `arxiv-papers` | OpenSearch index name |
| `OPENSEARCH__HOST` | `http://opensearch:9200` | OpenSearch cluster endpoint |
| `OLLAMA_MODEL` | `llama3.2:1b` | Local LLM model |

#### **Service-Aware Configuration**

The configuration system automatically detects the service context:
- **API Service**: Uses `localhost` for database and service connections
- **Airflow Service**: Uses Docker container hostnames (`postgres`, `opensearch`)

```python
# Configuration is automatically loaded based on context
from src.config import get_settings

settings = get_settings()  # Auto-detects API vs Airflow
print(f"ArXiv max results: {settings.arxiv.max_results}")
```

---

## 🔧 Reference & Development Guide

### **🛠️ Technology Stack**

| Service | Purpose | Status |
|---------|---------|--------|
| **FastAPI** | REST API with automatic docs | ✅ Ready |
| **PostgreSQL 16** | Paper metadata and content storage | ✅ Ready |
| **OpenSearch 2.19** | Hybrid search engine | ✅ Ready |
| **Apache Airflow 3.0** | Workflow automation | ✅ Ready |
| **Ollama** | Local LLM serving | ✅ Ready |

**Development Tools:** UV, Ruff, MyPy, Pytest, Docker Compose

### **🔧 Essential Commands**

#### **Using the Makefile** (Recommended)
```bash
# View all available commands
make help

# Quick workflow
make start         # Start all services
make health        # Check all services health
make test          # Run tests
make stop          # Stop services
```

#### **All Available Commands**
| Command | Description |
|---------|-------------|
| `make start` | Start all services |
| `make stop` | Stop all services |
| `make restart` | Restart all services |
| `make status` | Show service status |
| `make logs` | Show service logs |
| `make health` | Check all services health |
| `make setup` | Install Python dependencies |
| `make format` | Format code |
| `make lint` | Lint and type check |
| `make test` | Run tests |
| `make test-cov` | Run tests with coverage |
| `make clean` | Clean up everything |

#### **Direct Commands** (Alternative)
```bash
# If you prefer using commands directly
docker compose up --build -d    # Start services
docker compose ps               # Check status
docker compose logs            # View logs
uv run pytest                 # Run tests
```

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
- LLM context building runs after PDF parsing and stores `llm_summary`, `llm_key_points`, and `llm_context` in Postgres

### Airflow Pipeline Diagram
![Airflow ingestion flow](static/week2_data_ingestion_flow.png)


### Future Improvements (Context Building)
- Enforce strict JSON responses from Ollama to avoid markdown fallbacks
- Add a smaller, dedicated LLM input cap (separate from PDF max chars)
- Implement chunking + map-reduce summarization for long PDFs
- Add retries/backoff + partial results when LLM times out
- Cache LLM outputs to avoid recompute during re-ingestion
- Allow per-model timeout and context limits via env settings
- Add a lightweight/faster model option for quick summaries
