# SupportCopilot AI (Foundation MVP)

This is the production-grade **foundation** for SupportCopilot AI:

- FastAPI backend
- Ticket ingestion (`POST /upload-tickets`) for CSV/JSON
- Document ingestion (`POST /upload-docs`) for TXT/PDF with basic paragraph chunking
- Structured storage (raw uploads + processed JSON/text)
- API endpoints tested and working

No embeddings / RAG / LLM calls yet.

## Setup

From the `support-copilot-ai/` directory:

```bash
cd support-copilot-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

API docs:
- http://localhost:8000/docs

## API Usage Examples

### Health check

```bash
curl http://localhost:8000/health
```

### Upload tickets (CSV)

Sample CSV (`tickets.csv`):

```csv
id,query,response,timestamp
T1,How do I reset my password?,To reset your password go to Settings > Security,2026-01-01T00:00:00Z
T2,My subscription was charged twice?,Sorry about that—please verify your billing history,2026-01-02T00:00:00Z
```

Request:

```bash
curl -X POST http://localhost:8000/upload-tickets \
  -F "file=@tickets.csv"
```

### Upload tickets (JSON)

Sample JSON (`tickets.json`):

```json
{
  "tickets": [
    { "id": "T1", "query": "How do I reset my password?", "response": "To reset..." , "timestamp": "2026-01-01T00:00:00Z" }
  ]
}
```

Request:

```bash
curl -X POST http://localhost:8000/upload-tickets \
  -F "file=@tickets.json"
```

### Upload documents (TXT)

Request:

```bash
curl -X POST http://localhost:8000/upload-docs \
  -F "file=@kb.txt"
```

### Upload documents (PDF)

Request:

```bash
curl -X POST http://localhost:8000/upload-docs \
  -F "file=@kb.pdf"
```

## Where data is saved

- Raw uploads: `data/raw/`
- Processed outputs:
  - tickets: `data/processed/*.json`
  - docs: `data/processed/*.txt`

