#!/usr/bin/env python3
"""
Rebuild the vector index from scratch.

Usage:
    python scripts/rebuild_index.py                          # uses demo_data/sample_tickets.csv
    python scripts/rebuild_index.py --tickets path/to.csv    # custom tickets file
    python scripts/rebuild_index.py --docs demo_data/sample_kb.md  # also ingest a KB doc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app.*` imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.core.db import init_db, connect
from app.ingestion.ticket_loader import load_tickets
from app.ingestion.doc_parser import chunk_text, load_doc_text
from app.embeddings.vector_store import VectorStore
from app.utils.helpers import normalize_tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the SupportCopilot vector index.")
    parser.add_argument(
        "--tickets",
        default=str(PROJECT_ROOT / "demo_data" / "sample_tickets.csv"),
        help="Path to tickets file (CSV/JSON/JSONL). Default: demo_data/sample_tickets.csv",
    )
    parser.add_argument(
        "--docs",
        nargs="*",
        default=[],
        help="Optional paths to knowledge-base documents (.txt/.md/.pdf) to ingest.",
    )
    args = parser.parse_args()

    # ---------- Settings + DB ----------
    settings = get_settings()
    settings.resolve_paths()
    init_db()
    print(f"✓ Database initialized at {settings.db_path}")

    # ---------- Tickets ----------
    tickets_path = Path(args.tickets)
    if not tickets_path.exists():
        print(f"✗ Tickets file not found: {tickets_path}")
        sys.exit(1)

    tickets = load_tickets(str(tickets_path))

    with connect() as conn:
        conn.execute("DELETE FROM feedback;")
        conn.execute("DELETE FROM generations;")
        conn.execute("DELETE FROM vector_items WHERE source_type = 'ticket';")
        conn.execute("DELETE FROM tickets;")
        conn.commit()

        for t in tickets:
            tags = normalize_tags(t.tags)
            tags_json = json.dumps(tags)
            conn.execute(
                """
                INSERT INTO tickets(ticket_id, customer_query, historical_response, timestamp, tags)
                VALUES (?, ?, ?, ?, ?)
                """,
                (t.ticket_id, t.customer_query, t.response, t.timestamp, tags_json),
            )
            vector_text = (
                f"Ticket ID: {t.ticket_id}\n"
                f"Customer Query: {t.customer_query}\n"
                f"Historical Response: {t.response}\n"
            )
            conn.execute(
                """
                INSERT INTO vector_items(source_type, source_ref, text, tags)
                VALUES (?, ?, ?, ?)
                """,
                ("ticket", t.ticket_id, vector_text, tags_json),
            )
        conn.commit()

    print(f"✓ Ingested {len(tickets)} tickets from {tickets_path.name}")

    # ---------- Docs (optional) ----------
    chunks_added = 0
    for doc_path_str in args.docs:
        doc_path = Path(doc_path_str)
        if not doc_path.exists():
            print(f"  ⚠ Doc not found, skipping: {doc_path}")
            continue
        text = load_doc_text(str(doc_path))
        chunks = chunk_text(
            text,
            chunk_size_chars=settings.chunk_size_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
        with connect() as conn:
            for idx, chunk in enumerate(chunks):
                source_ref = f"{doc_path.name}#{idx}"
                vector_text = f"Doc Excerpt ({doc_path.name}) [chunk {idx}]: {chunk}\n"
                conn.execute(
                    """
                    INSERT INTO vector_items(source_type, source_ref, text, tags)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("doc", source_ref, vector_text, json.dumps([])),
                )
                chunks_added += 1
            conn.commit()
        print(f"  ✓ Ingested {len(chunks)} chunks from {doc_path.name}")

    # ---------- Rebuild vector index ----------
    vs = VectorStore()
    count = vs.rebuild_from_db()
    print(f"✓ Vector index rebuilt — {count} items indexed")
    print("\nDone! Start the server with:  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
