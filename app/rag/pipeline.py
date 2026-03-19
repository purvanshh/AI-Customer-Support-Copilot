from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.db import connect
from app.core.config import get_settings
from app.ingestion.ticket_loader import load_tickets
from app.ingestion.doc_parser import chunk_text, load_doc_text
from app.embeddings.vector_store import VectorStore
from app.rag.retriever import retrieve_context
from app.rag.generator import generate_response
from app.models.classifier import classify_ticket
from app.api.schemas import AnalyticsResponse, FeedbackResponse, FeedbackRequest, QueryRequest, QueryResponse, UploadDocsResponse, UploadTicketsResponse
from app.utils.helpers import normalize_tags, now_iso
from app.core.logger import configure_logger
from fastapi import HTTPException

logger = configure_logger()


async def upload_tickets(tmp_path: str, original_filename: str | None = None) -> UploadTicketsResponse:
    settings = get_settings()
    tickets = load_tickets(tmp_path)

    vector_items_added = 0
    with connect() as conn:
        # Reset corpus + generations for a clean MVP demo.
        conn.execute("DELETE FROM feedback;")
        conn.execute("DELETE FROM generations;")
        # Keep knowledge-base documents; only remove ticket-derived vectors.
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

            # One vector item per ticket (enough for MVP).
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
            vector_items_added += 1

        conn.commit()

    vs = VectorStore()
    vs.rebuild_from_db()

    return UploadTicketsResponse(
        tickets_ingested=len(tickets),
        vector_items_added=vector_items_added,
        index_rebuilt=True,
    )


async def upload_docs(doc_paths: list[tuple[str, str]]) -> UploadDocsResponse:
    """
    Args:
      doc_paths: list of (tmp_path, original_filename)
    """
    settings = get_settings()

    docs_processed = 0
    chunks_added = 0

    with connect() as conn:
        for tmp_path, original_filename in doc_paths:
            docs_processed += 1
            text = load_doc_text(tmp_path)
            chunks = chunk_text(
                text,
                chunk_size_chars=settings.chunk_size_chars,
                overlap_chars=settings.chunk_overlap_chars,
            )
            for idx, chunk in enumerate(chunks):
                source_ref = f"{original_filename}#{idx}"
                vector_text = f"Doc Excerpt ({original_filename}) [chunk {idx}]: {chunk}\n"
                tags_json = json.dumps([])
                conn.execute(
                    """
                    INSERT INTO vector_items(source_type, source_ref, text, tags)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("doc", source_ref, vector_text, tags_json),
                )
                chunks_added += 1

        conn.commit()

    vs = VectorStore()
    vs.rebuild_from_db()

    return UploadDocsResponse(docs_processed=docs_processed, chunks_added=chunks_added, index_rebuilt=True)


async def query_support(req: QueryRequest) -> QueryResponse:
    settings = get_settings()
    top_k = int(req.top_k or settings.top_k_default)

    classification = classify_ticket(req.customer_query)
    inferred_tag = classification["label"]
    tags_final = req.tags if req.tags else [inferred_tag]
    allowed_tags = req.tags

    try:
        context_text, sources = retrieve_context(
            req.customer_query,
            top_k=top_k,
            allowed_tags=allowed_tags,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    gen = generate_response(
        customer_query=req.customer_query,
        context=context_text,
        sources=sources,
        tags=tags_final,
        llm_backend_override=req.llm_override,
    )

    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO generations(customer_query, suggestion_text, confidence, model_used, response_time_ms, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
            """,
            (
                req.customer_query,
                gen["suggestion_text"],
                float(gen["confidence"]),
                str(gen["model_used"]),
                int(gen["response_time_ms"]),
            ),
        )
        generation_id = int(cur.lastrowid)
        conn.commit()

    # Structured query log — query, classification, model, latency, confidence
    logger.info(
        "QUERY gen_id=%d | category=%s | model=%s | latency_ms=%d | confidence=%.2f | query=%s",
        generation_id,
        classification["label"],
        gen["model_used"],
        gen["response_time_ms"],
        gen["confidence"],
        req.customer_query[:120],
    )

    # Convert sources for response
    resp_sources = []
    for s in sources[:top_k]:
        resp_sources.append(
            {
                "source_type": s["source_type"],
                "source_ref": s["source_ref"],
                "tags": s.get("tags"),
                "score": float(s.get("score", 0.0)),
            }
        )

    return QueryResponse(
        generation_id=generation_id,
        suggestion_text=gen["suggestion_text"],
        confidence=float(gen["confidence"]),
        sources=resp_sources,
        model_used=gen["model_used"],
    )


async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    action = req.user_action

    corrected_text = req.corrected_text if action == "corrected" else None

    with connect() as conn:
        # Update generation status first.
        new_status = action
        conn.execute(
            """
            UPDATE generations
            SET status = ?
            WHERE id = ?
            """,
            (new_status, req.generation_id),
        )

        if corrected_text:
            conn.execute(
                """
                UPDATE generations
                SET suggestion_text = ?
                WHERE id = ?
                """,
                (corrected_text, req.generation_id),
            )

        # Replace any prior feedback (unique per generation).
        conn.execute("DELETE FROM feedback WHERE generation_id = ?;", (req.generation_id,))
        conn.execute(
            """
            INSERT INTO feedback(generation_id, user_action, corrected_text)
            VALUES (?, ?, ?)
            """,
            (req.generation_id, action, corrected_text),
        )

        conn.commit()

    return FeedbackResponse(status="ok")


async def get_analytics() -> AnalyticsResponse:
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM generations;").fetchone()["c"]
        accepted = conn.execute("SELECT COUNT(*) AS c FROM generations WHERE status = 'accepted';").fetchone()["c"]
        avg_time = conn.execute("SELECT AVG(response_time_ms) AS a FROM generations;").fetchone()["a"]
        feedback_count = conn.execute("SELECT COUNT(*) AS c FROM feedback;").fetchone()["c"]

        correct_count = conn.execute(
            """
            SELECT COUNT(*) AS c FROM feedback
            WHERE user_action IN ('accepted', 'corrected')
            """
        ).fetchone()["c"]

    total_f = int(total or 0)
    feedback_count_i = int(feedback_count or 0)
    accepted_i = int(accepted or 0)
    avg_time_i = float(avg_time) if avg_time is not None else 0.0
    accuracy = (correct_count / feedback_count_i) if feedback_count_i > 0 else 0.0
    percent_auto_resolved = (accepted_i / total_f) * 100.0 if total_f > 0 else 0.0

    return AnalyticsResponse(
        tickets_processed=total_f,
        percent_auto_resolved=percent_auto_resolved,
        avg_response_time_ms=avg_time_i,
        accuracy=float(accuracy),
        feedback_count=feedback_count_i,
    )

