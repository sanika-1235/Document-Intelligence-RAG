# DDRAG

**Drag, Drop, Retrieve, Augment, Generate**

DDRAG is a local-first retrieval-augmented generation (RAG) application for asking questions about your own documents. It accepts PDF, DOCX, Markdown, and text files, retrieves relevant passages with PostgreSQL and pgvector, and generates answers locally through Ollama with source attribution.

## Demo

🎥 **[Watch the DDRAG demo on YouTube](https://youtu.be/F0H7eqERpBA)**

The video demonstrates account creation, document ingestion, semantic retrieval, answers with source attribution, and persistent chat history.

DDRAG runs locally with Docker, PostgreSQL + pgvector, and Ollama, keeping documents and model inference on the local machine.

### Interface preview

<p align="center">
  <img src="docs/images/desktop-chat.png" alt="DDRAG desktop chat showing an answer, source attribution, and chat history" width="100%">
</p>

*An example response with retrieved source metadata and a persisted chat session; source display alone does not establish answer grounding.*

<p align="center">
  <img src="docs/images/mobile-chat.png" alt="DDRAG responsive mobile chat interface" width="34%">
</p>

*The chat workflow adapted for a narrow viewport.*

## Features

- JWT authentication with Argon2id password hashing and user-owned resources
- PDF, DOCX, Markdown, and TXT ingestion with file validation and duplicate detection
- deterministic, overlapping text chunks with versioned chunking metadata
- local embeddings and answer generation through Ollama
- owner-scoped cosine-distance retrieval with PostgreSQL + pgvector
- grounded prompts, source citations, and persistent chat history
- responsive React interface

## How It Works

```text
Upload document → Extract text → Create overlapping chunks → Generate embeddings
       → Store in PostgreSQL + pgvector → Retrieve relevant owned chunks
       → Build grounded context → Generate locally → Return answer + sources
```

The backend treats retrieved text as reference material and instructs the model to answer from the supplied context. When retrieval returns no usable context, DDRAG returns an insufficient-information response without calling the generation model. Retrieval has no relevance threshold, so a non-empty result does not establish that the question is answerable.

## Architecture

```text
Browser / React + Vite
          │ REST
          ▼
       FastAPI
          │
   ┌──────┼──────────────┐
   ▼      ▼              ▼
 Auth  Documents       Chats
          │              │
          ├── local document storage
          ├── PostgreSQL + pgvector
          └── Ollama (embeddings + generation)
```

FastAPI handles authentication, authorization, ingestion, retrieval, generation, and persistence. The React frontend consumes the backend API. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for request flows, the data model, security boundaries, and detailed tradeoffs.

## Tech Stack

| Area | Technologies |
| --- | --- |
| Frontend | React, TypeScript, Vite, Chakra UI |
| Backend | Python 3.13, FastAPI, SQLAlchemy, Alembic |
| RAG | Ollama, vector embeddings, cosine-distance retrieval, grounded prompting |
| Data | PostgreSQL 16, pgvector, local document storage |
| Runtime | Docker, Docker Compose |
| Testing | pytest, dedicated PostgreSQL test database |

## Quick Start

Prerequisites: Git, Docker with Docker Compose, Ollama, and Node.js with npm. Python 3.13 is only required when running the backend outside Docker.

Clone the repository:

```bash
git clone https://github.com/drZania/ddrag.git
cd ddrag
```

If the backend `.env` file is absent, create it with `if (-not (Test-Path .env)) { Copy-Item .env.example .env }` in PowerShell or `test -f .env || cp .env.example .env` on Linux/macOS. Preserve any existing `.env`. Replace the checked-in `JWT_SECRET` placeholder in a newly created file with a unique local value.

Pull the local models and start the database and backend:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen2.5:1.5b
docker compose up -d
```

In a second terminal, prepare and start the frontend:

```bash
cd frontend
npm ci
# PowerShell: if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Linux/macOS: test -f .env || cp .env.example .env
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`. The backend applies Alembic migrations on startup and exposes `http://localhost:8000/health`.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed Windows, Linux, and macOS setup, backend development, test database setup, reset behavior, and troubleshooting.

## Testing

Backend tests require `TEST_DATABASE_URL` to point to a dedicated PostgreSQL database whose name contains `test`. Never point it at the development database. See [Running Tests](docs/DEVELOPMENT.md#14-running-tests) before running `pytest`.

Frontend checks:

```bash
cd frontend
npm run lint
npm run typecheck
npm run build
```

## Evaluation

DDRAG evaluates retrieval, answer correctness, grounding, refusal behavior, operational reliability, and latency as separate dimensions. It does not combine them into one overall accuracy score.

The evaluation uses a fixed 40-question Caraga State University (CSU) Student Handbook benchmark: 34 answerable, 3 ambiguous, and 3 unanswerable questions. The source PDF and handbook-derived benchmark are not redistributed; reproducing the study requires authorized copies matching the published [benchmark metadata](evaluation/benchmarks/README.md) and [corpus identity](evaluation/corpus/README.md).

Historical baseline retrieval results over the 37 retrieval-eligible questions were:

| Metric | Result |
| --- | ---: |
| Hit@5 | 0.973 |
| Recall@5 | 0.851 |
| CompleteEvidence@5 | 0.730 |

These are retrieval measures, not overall answer accuracy. Complete evidence retrieval remained weaker than finding at least one required evidence unit, and generation incompleteness could remain even when retrieval was complete.

Under the frozen benchmark and tested local setup, the paired generation-model experiment did not support adopting `qwen2.5:3b`; DDRAG retained `qwen2.5:1.5b`. The `ddrag-answer-completeness-v1` prompt produced some useful refusal improvements but did not improve the predeclared complete-retrieval generation-failure population, so `ddrag-production-v1` was retained.

The retained tested configuration is:

| Component | Value |
| --- | --- |
| Generation | `qwen2.5:1.5b` |
| Prompt | `ddrag-production-v1` |
| Embedding | `qwen3-embedding:0.6b` |
| Retrieval | owner-scoped pgvector cosine, top-K 5 |
| Chunking | 1000 characters with 200-character overlap |

Only the generation model and generation prompt received completed controlled comparisons. The embedding model, top-K, chunk size, chunk overlap, and retrieval strategy were retained for continuity and were not established as optimal.

Qualitative judgments used blinded AI-assisted review and independent adjudication; they complement deterministic metrics and do not constitute an objective ground truth. Known limitations include incomplete multi-chunk and summarization coverage, substantial local generation latency, one corpus and 40-question benchmark, and private inputs required to reproduce the study. See the [evaluation article](evaluation/reports/ddrag_evaluation_article.md), [evaluation guide](evaluation/README.md), and [frozen evidence report](evaluation/reports/ddrag_evaluation_report.md).

## Design Highlights

- **PostgreSQL + pgvector** keeps relational ownership data, document chunks, and embeddings in one transactional data store.
- **Local inference through Ollama** keeps embedding and generation requests on the local machine.
- **Grounded source attribution** persists the retrieved chunks used for an answer so the interface can display backend-provided sources.
- **Backend-enforced ownership** derives resource access from the authenticated user and scopes document, retrieval, and chat queries server-side.

See [Key Design Decisions](docs/ARCHITECTURE.md#21-key-design-decisions) for detailed tradeoffs and architecture discussion.

## Project Structure

```text
app/                    FastAPI application and RAG pipeline
frontend/src/           React interface and API client
migrations/             Alembic schema history
tests/                  Backend tests and database safety setup
evaluation/             Evaluation framework, metadata, selected evidence, and reports
docs/ARCHITECTURE.md    System design and request flows
docs/DEVELOPMENT.md     Detailed local development guide
```

## License

MIT License. See [LICENSE](LICENSE).
