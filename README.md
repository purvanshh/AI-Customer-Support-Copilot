# SupportCopilot AI (MVP)

FastAPI backend that ingests historical support tickets, builds embeddings + vector search, and exposes:

- `POST /upload-tickets` (CSV/JSON)
- `POST /query` (suggest a support reply via RAG)
- `POST /feedback` (human-in-the-loop)
- `GET /analytics` (basic metrics)

## Quickstart

1. Create and activate a virtual environment.
2. Install deps:

```bash
pip install -r requirements.txt
```

Optional (recommended for better retrieval + real LLMs):

```bash
pip install -r requirements-optional.txt
```

3. Copy env:

```bash
cp .env.example .env
```

4. Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

API docs are available at `http://localhost:8000/docs`.

## Data formats

### CSV (recommended)

CSV columns (case-insensitive, underscores/spaces allowed):
- `Ticket ID`
- `Customer query`
- `Response`
- `Timestamp` (optional)
- `Tags` (optional, comma-separated)

### JSON

Either:
- a JSON array of ticket objects, or
- `{ "tickets": [...] }`

Each ticket object should contain fields compatible with the CSV mappings.

## Notes

By default this MVP runs with `LLM_BACKEND=mock` and `EMBEDDING_BACKEND=mock` so you can test the full pipeline without API keys. Switch to `sentence_transformers` and `openai`/`gemini` when you’re ready.

