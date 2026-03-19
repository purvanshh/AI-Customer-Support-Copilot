from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.models.classifier import CATEGORIES
from app.services.classification_service import ClassificationService
from app.storage.db import get_conn, init_db


class AnalyticsService:
    """
    Compute analytics derived from stored feedback.
    """

    def __init__(self) -> None:
        init_db()
        self.classifier = ClassificationService()

    def compute(self) -> dict[str, Any]:
        """
        Compute analytics metrics from feedback table.

        Returns:
            Dict matching the /analytics API contract.
        """
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT query, rating
                FROM feedback
                WHERE rating IS NOT NULL
                """
            ).fetchall()

        total = len(rows)
        if total == 0:
            return {
                "total_queries": 0,
                "avg_rating": 0.0,
                "resolution_rate": 0.0,
                "top_category": "general",
                "feedback_count": 0,
            }

        ratings = [int(r["rating"]) for r in rows if r["rating"] is not None]
        avg_rating = sum(ratings) / max(1, len(ratings))
        high_rated = [r for r in ratings if r >= 4]
        resolution_rate = len(high_rated) / total

        category_counts: Counter[str] = Counter()
        for r in rows:
            label = self.classifier.classify(str(r["query"])).label
            if label not in CATEGORIES:
                label = "general"
            category_counts[label] += 1

        top_category = category_counts.most_common(1)[0][0] if category_counts else "general"

        return {
            "total_queries": total,
            "avg_rating": float(avg_rating),
            "resolution_rate": float(resolution_rate),
            "top_category": top_category,
            "feedback_count": total,
        }

