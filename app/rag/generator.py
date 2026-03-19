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
    m = re.search(r"Historical Response:\s*(.+)", context, flags=re.IGNORECASE | re.DOTALL)
    snippet = m.group(1).strip() if m else ""
    snippet = snippet.split("\n")[0].strip() if snippet else ""
    if not snippet:
        # If we couldn't find a historical agent response (e.g., doc-only context),
        # use the first meaningful context line as a grounding hint.
        for line in context.splitlines():
            if line.strip():
                snippet = line.strip()
                break
        snippet = snippet[:300] if snippet else ""
        if not snippet:
            snippet = "Thanks for reaching out. Could you share a bit more detail about your setup so we can pinpoint the issue?"
    return (
        f"Thanks for reaching out.\n\n"
        f"Here’s what has helped in similar cases: {snippet}\n\n"
        "If you reply with any extra context (account email, steps tried, and any error messages), "
        "we’ll help you get unblocked."
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

