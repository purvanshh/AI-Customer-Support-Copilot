from __future__ import annotations

from typing import Any


CATEGORIES = ["Billing", "Technical issue", "Refund", "General query"]


def classify_ticket(customer_query: str) -> dict[str, Any]:
    """
    Lightweight keyword classifier to keep the MVP dependency-free.
    Returns: {label, confidence, matched_keywords}
    """
    q = customer_query.lower()

    rules = {
        "Billing": ["invoice", "billing", "charge", "payment", "receipt", "plan", "upgrade", "downgrade"],
        "Technical issue": ["error", "bug", "issue", "doesn't", "cannot", "failed", "crash", "login", "timeout", "broken"],
        "Refund": ["refund", "chargeback", "cancel", "cancellation", "money back", "returned"],
        "General query": ["how", "what", "where", "pricing", "features", "support", "help", "question"],
    }

    best_label = "General query"
    best_score = 0.0
    best_keywords: list[str] = []

    for label, keywords in rules.items():
        hits = [k for k in keywords if k in q]
        score = len(hits) / max(1, len(keywords))
        if score > best_score:
            best_score = score
            best_label = label
            best_keywords = hits

    # Map heuristic score to confidence in [0.2, 0.95] for usability.
    confidence = float(min(0.95, max(0.2, best_score * 5))) if best_keywords else 0.3
    return {"label": best_label, "confidence": confidence, "matched_keywords": best_keywords}

