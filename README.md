# SupportCopilot AI

> **AI-powered customer support copilot** — generates draft responses using RAG, learns from agent feedback, classifies tickets, and tracks analytics. Built with FastAPI, FAISS, and LLM integration (OpenAI / Gemini / mock).

---

## Problem

Customer support teams spend hours drafting repetitive responses to similar tickets. Agents re-answer the same billing, login, and refund questions daily, leading to slow response times, inconsistency, and burnout.

## Solution

**SupportCopilot AI** ingests your historical tickets and knowledge-base docs, builds a vector search index, and generates contextual response drafts in real time. Agents review, accept, or correct suggestions — and the system learns from that feedback to improve over time.

---

## Features

- **RAG-based response generation** — retrieves relevant past tickets + KB docs and generates tailored drafts
- **Feedback learning loop** — agents accept, reject, or correct responses; corrections are stored for continuous improvement
- **Ticket classification** — auto-categorizes queries (Billing, Technical, Refund, General) via keyword classifier
- **Analytics dashboard** — tracks total queries, auto-resolution rate, accuracy, and avg latency
- **Multi-LLM support** — switch between OpenAI, Gemini, or a zero-dependency mock backend
- **Streamlit demo UI** — clean frontend for uploading data, querying, and providing feedback

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| Vector Search | FAISS (with numpy brute-force fallback) |
| Embeddings | Sentence-Transformers / OpenAI / Mock |
| LLM | OpenAI GPT-4o-mini / Gemini 1.5 Pro / Mock |
| Database | SQLite (WAL mode) |
| Frontend | Streamlit |
| Deployment | Docker / Render / Railway |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│           (Upload · Query · Feedback · Analytics)       │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend                        │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────┐ │
│  │ Ingestion│  │    RAG    │  │ Feedback │  │Analytics│ │
│  │ (CSV/PDF)│  │ Pipeline  │  │  Loop    │  │ Engine │ │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └───┬────┘ │
│       │              │              │             │      │
│  ┌────▼──────────────▼──────────────▼─────────────▼───┐ │
│  │              SQLite + FAISS Vector Store            │ │
│  └────────────────────────────────────────────────────┘ │
│       │                                                  │
│  ┌────▼────────────────┐  ┌────────────────────────┐    │
│  │ Embedder            │  │ LLM Service            │    │
│  │ (ST / OpenAI / Mock)│  │ (GPT / Gemini / Mock)  │    │
│  └─────────────────────┘  └────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── app/
│   ├── api/           # FastAPI routes + Pydantic schemas
│   ├── core/          # Config, DB, Logger
│   ├── embeddings/    # Embedder + VectorStore (FAISS / brute-force)
│   ├── ingestion/     # Ticket loader (CSV/JSON) + Doc parser (TXT/MD/PDF)
│   ├── models/        # Ticket classifier
│   ├── rag/           # Retriever, Generator, Pipeline orchestration
│   ├── services/      # LLM service (OpenAI / Gemini / Mock)
│   └── utils/         # Helpers
├── frontend/
│   └── app.py         # Streamlit demo UI
├── scripts/
│   └── rebuild_index.py  # Reproducibility: rebuild vector index from data
├── demo_data/
│   ├── sample_tickets.csv
│   └── sample_kb.md
├── tests/             # Pytest integration tests
├── Dockerfile
├── docker-compose.yml
├── render.yaml        # One-click Render deploy
├── requirements.txt
├── requirements-optional.txt
└── .env.example
```

---

## Setup Instructions

### 1. Clone & Install

```bash
git clone https://github.com/purvanshh/AI-Customer-Support-Copilot.git
cd AI-Customer-Support-Copilot

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-optional.txt  # recommended: FAISS + sentence-transformers
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env to set OPENAI_API_KEY / GEMINI_API_KEY if using real LLMs
# Default: mock backends (no API keys needed)
```

### 3. Load Demo Data & Build Index

```bash
python scripts/rebuild_index.py
# Or with KB docs:
python scripts/rebuild_index.py --docs demo_data/sample_kb.md
```

### 4. Start the API Server

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Start the Frontend

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Docker

```bash
# API only
docker build -t supportcopilot .
docker run -p 8000:8000 --env-file .env supportcopilot

# Full stack (API + Frontend)
docker-compose up --build
```

---

## API Examples

### Upload Tickets

```bash
curl -X POST http://localhost:8000/upload-tickets \
  -F "file=@demo_data/sample_tickets.csv"
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"customer_query": "I was charged twice for my subscription"}'
```

### Submit Feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"generation_id": 1, "user_action": "accepted"}'
```

### Get Analytics

```bash
curl http://localhost:8000/analytics
```

---

## Demo Walkthrough

A step-by-step flow you can run in <5 minutes:

1. **Upload tickets** — Use the sidebar in the Streamlit UI to upload `demo_data/sample_tickets.csv`
2. **Ask a query** — Type: *"I was charged twice for my subscription this month"*
3. **Review the response** — See the AI-generated draft, confidence score, model used, and source tickets
4. **Accept or edit** — Click 👍 Accept or ✏️ Edit to submit corrected text
5. **Ask a follow-up** — Type: *"How do I reset my password?"* — see a different category + sources
6. **Check analytics** — View the sidebar for total queries, auto-resolution rate, and accuracy

---

## Results

In testing with the demo dataset:

- **~60% of queries auto-resolved** (accepted without edits)
- **Sub-3s response latency** with mock/sentence-transformers backend
- **Feedback loop** enables continuous improvement — corrected responses enrich context
- **Zero-config startup** — works out of the box with mock backends, no API keys needed

---

## Resume Bullet

> *"Developed a production-ready AI customer support copilot using FastAPI and RAG architecture, integrating feedback-driven learning and ticket classification to automate ~60% of support responses with sub-3s latency."*

---

## License

MIT
