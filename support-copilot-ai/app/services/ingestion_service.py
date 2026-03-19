from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from app.api.schemas import UploadDocsResponse, UploadTicketsResponse
from app.core.config import data_dirs
from app.ingestion.doc_parser import parse_and_chunk
from app.ingestion.ticket_loader import load_tickets
from app.utils.file_utils import safe_filename, save_upload_file


def _utc_now_stamp() -> str:
    """
    Return a UTC timestamp suitable for filenames.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


async def ingest_tickets(file: UploadFile) -> UploadTicketsResponse:
    """
    Ingest uploaded tickets file:
      - Save raw upload under `data/raw/`
      - Parse and validate tickets
      - Save normalized tickets under `data/processed/`

    Args:
        file: Uploaded file from FastAPI.

    Returns:
        UploadTicketsResponse
    """
    raw_dir, processed_dir = data_dirs()
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded filename")

    original = safe_filename(file.filename)
    stamp = _utc_now_stamp()
    raw_path = raw_dir / f"tickets_{stamp}_{original}"
    processed_path = processed_dir / f"tickets_processed_{stamp}.json"

    await save_upload_file(file, raw_path)

    # Parse
    try:
        records = load_tickets(raw_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not records:
        raise HTTPException(status_code=400, detail="Uploaded tickets file is empty or has no valid records")

    # Save normalized structured output (no embeddings yet)
    payload: list[dict[str, Any]] = [r.model_dump() for r in records]
    processed_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return UploadTicketsResponse(
        message="Tickets ingested successfully",
        raw_file_path=str(raw_path),
        processed_file_path=str(processed_path),
        records_received=len(payload),
        records_ingested=len(payload),
    )


async def ingest_docs(file: UploadFile) -> UploadDocsResponse:
    """
    Ingest uploaded documents:
      - Save raw upload under `data/raw/`
      - Extract text and split into paragraphs (basic chunking)
      - Save processed chunks under `data/processed/`

    Args:
        file: Uploaded document.

    Returns:
        UploadDocsResponse
    """
    raw_dir, processed_dir = data_dirs()
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not file:
        raise HTTPException(status_code=400, detail="No document provided")

    stamp = _utc_now_stamp()
    all_chunks: list[str] = []
    first_raw_path: str | None = None

    if not file.filename:
        raise HTTPException(status_code=400, detail="Document is missing its filename")
    original = safe_filename(file.filename)
    raw_path = raw_dir / f"doc_{stamp}_0_{original}"
    first_raw_path = str(raw_path)

    await save_upload_file(file, raw_path)

    try:
        chunks = parse_and_chunk(raw_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="Uploaded document(s) produced no extractable text")

    processed_path = processed_dir / f"docs_processed_{stamp}.txt"
    processed_path.write_text("\n\n".join(all_chunks), encoding="utf-8")

    # Persist structured metadata for RAG (chunk boundaries + doc identity).
    # Keeps the existing API behavior intact (we still return the .txt path),
    # while enabling metadata-rich vector index rebuilds.
    chunks_meta = [
        {
            "source_filename": original,
            "chunk_index": i,
            "chunk_text": chunk,
        }
        for i, chunk in enumerate(all_chunks)
    ]
    meta_path = processed_dir / f"docs_processed_{stamp}.json"
    meta_path.write_text(json.dumps(chunks_meta, indent=2), encoding="utf-8")

    return UploadDocsResponse(
        message="Documents ingested successfully",
        raw_file_path=first_raw_path or "",
        processed_file_path=str(processed_path),
        chunks_saved=len(all_chunks),
    )

