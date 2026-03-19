from __future__ import annotations

import time
import re
from typing import Any

import numpy as np

from app.services.llm_service import LLMService


def _score_to_confidence(score: float) -> float:
    # FAISS IP over normalized embeddings approximates cosine similarity in [-1, 1].
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


def _fallback_response(customer_query: str, context: str) -> str:
    """
    Grounded fallback: extract the most useful information from the context
    without depending on the LLM.
    """
    # ── 1. Prefer feedback-corrected responses ──
    for m in re.finditer(r"Corrected Response:\s*(.+?)(?=\n\[|\Z)", context, flags=re.DOTALL):
        snippet = m.group(1).strip().split("\n")[0].strip()
        if snippet:
            return (
                f"Thanks for reaching out.\n\n"
                f"{snippet}\n\n"
                "Let us know if you need anything else."
            )

    # ── 2. Try historical ticket responses ──
    for m in re.finditer(r"Historical Response:\s*(.+?)(?=\nTicket ID:|\n\[|\Z)", context, flags=re.DOTALL):
        snippet = m.group(1).strip().split("\n")[0].strip()
        if snippet:
            return (
                f"Thanks for reaching out.\n\n"
                f"Based on similar past cases: {snippet}\n\n"
                "If you need further assistance, please share your account details."
            )

    # ── 3. Try doc excerpts ──
    for line in context.splitlines():
        line = line.strip()
        if line and not line.startswith("[") and len(line) > 30:
            return (
                f"Thanks for reaching out.\n\n"
                f"From our knowledge base: {line[:300]}\n\n"
                "Let us know if you need more details."
            )

    # ── 4. Absolute fallback ──
    return (
        "Thanks for reaching out.\n\n"
        "I don't have enough information in our knowledge base to answer this fully. "
        "Could you share more details (account email, steps tried, and any error messages) "
        "so we can assist you directly?"
    )


def generate_response(
    *,
    customer_query: str,
    context: str,
    sources: list[dict[str, Any]],
    tags: list[str] | None = None,
    llm_backend_override: str | None = None,
) -> dict[str, Any]:
    # Confidence is derived from retriever similarity.
    scores = [float(s.get("score", 0.0)) for s in sources[:5]]
    avg_score = float(np.mean(scores)) if scores else 0.0
    confidence = _score_to_confidence(avg_score)

    llm = LLMService()
    t0 = time.time()
    try:
        suggestion_text, meta = llm.generate(
            customer_query=customer_query,
            context=context,
            tags=tags,
            llm_backend_override=llm_backend_override,
        )
    except Exception:
        suggestion_text = _fallback_response(customer_query=customer_query, context=context)
        meta = {"model_used": (llm_backend_override or "mock"), "backend": "fallback", "response_time_ms": None}

    response_time_ms = int((time.time() - t0) * 1000)
    return {
        "suggestion_text": suggestion_text,
        "confidence": confidence,
        "model_used": meta.get("model_used", "unknown"),
        "backend": meta.get("backend", "unknown"),
        "response_time_ms": response_time_ms,
    }
