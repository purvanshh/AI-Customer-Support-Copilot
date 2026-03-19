from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


CATEGORIES: list[str] = ["billing", "technical", "refund", "general"]


@dataclass(frozen=True)
class ClassificationResult:
    """
    Classification output for a ticket/query.
    """

    label: str
    confidence: float
    matched_keywords: list[str]


def classify_keywords(query: str) -> ClassificationResult:
    """
    Lightweight keyword-based classifier for the MVP.

    This keeps classification deterministic and dependency-free while still
    enabling analytics and routing decisions.

    Args:
        query: Input ticket text.

    Returns:
        ClassificationResult with label + confidence.
    """
    q = (query or "").lower()
    rules: Dict[str, List[str]] = {
        "billing": [
            "invoice",
            "billing",
            "charge",
            "payment",
            "plan",
            "receipt",
            "subscription",
            "upgrade",
            "downgrade",
        ],
        "technical": [
            "error",
            "issue",
            "bug",
            "doesn't",
            "cannot",
            "failed",
            "crash",
            "login",
            "password",
            "reset password",
            "timeout",
            "broken",
            "not working",
            "app",
        ],
        "refund": [
            "refund",
            "chargeback",
            "cancel",
            "cancellation",
            "money back",
            "returned",
            "terminate",
        ],
        "general": [
            "how",
            "what",
            "why",
            "where",
            "pricing",
            "features",
            "support",
            "help",
            "question",
        ],
    }

    best_label = "general"
    best_score = 0.0
    best_hits: list[str] = []

    for label, keywords in rules.items():
        hits = [k for k in keywords if k in q]
        score = len(hits) / max(1, len(keywords))
        if score > best_score:
            best_score = score
            best_label = label
            best_hits = hits

    if best_hits:
        confidence = min(0.95, max(0.25, best_score * 5.0))
    else:
        confidence = 0.25

    return ClassificationResult(label=best_label, confidence=float(confidence), matched_keywords=best_hits)

