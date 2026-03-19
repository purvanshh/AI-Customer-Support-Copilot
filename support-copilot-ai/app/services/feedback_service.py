from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Optional

from app.storage.db import get_conn, init_db


@dataclass(frozen=True)
class FeedbackMatch:
    """
    Best feedback match for a given incoming query.
    """

    corrected_response: str
    ai_response: str
    rating: int
    similarity: float


class FeedbackService:
    """
    Persists and retrieves human feedback to make the RAG system adaptive.
    """

    def __init__(self) -> None:
        init_db()

    def save_feedback(
        self,
        *,
        query: str,
        ai_response: str,
        corrected_response: str,
        rating: int,
    ) -> int:
        """
        Store human feedback in the database.

        Args:
            query: Incoming customer query.
            ai_response: Model output that the user saw.
            corrected_response: User-edited/corrected response (can be same as ai_response).
            rating: Rating from 1-5.

        Returns:
            Inserted feedback id.
        """
        with get_conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO feedback(query, ai_response, corrected_response, rating)
                VALUES (?, ?, ?, ?)
                """,
                (query, ai_response, corrected_response, int(rating)),
            )
            conn.commit()
            return int(cur.lastrowid)

    def _similarity(self, a: str, b: str) -> float:
        """
        Compute a lightweight similarity score between two strings.
        """
        a_norm = " ".join((a or "").lower().split())
        b_norm = " ".join((b or "").lower().split())
        if not a_norm or not b_norm:
            return 0.0

        # SequenceMatcher ratio is simple, deterministic, and dependency-free.
        return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()

    def find_best_match(
        self,
        query: str,
        *,
        min_similarity: float = 0.78,
        min_rating_for_override: int = 4,
    ) -> Optional[FeedbackMatch]:
        """
        Find the most similar past query that has a sufficiently high rating.

        Args:
            query: Incoming query.
            min_similarity: Minimum similarity threshold [0,1].
            min_rating_for_override: Only consider feedback with rating>=this.

        Returns:
            FeedbackMatch if found, else None.
        """
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT query, ai_response, corrected_response, rating
                FROM feedback
                WHERE rating >= ?
                ORDER BY rating DESC, created_at DESC
                """,
                (int(min_rating_for_override),),
            ).fetchall()

        best: tuple[float, Optional[FeedbackMatch]] = (0.0, None)
        for r in rows:
            past_query = r["query"]
            sim = self._similarity(query, past_query)
            if sim < min_similarity:
                continue
            corrected = r["corrected_response"] or ""
            # Prioritize corrected response if present; otherwise fall back to ai_response.
            candidate_text = corrected.strip() or r["ai_response"]
            match = FeedbackMatch(
                corrected_response=candidate_text,
                ai_response=r["ai_response"],
                rating=int(r["rating"]),
                similarity=float(sim),
            )
            if sim > best[0]:
                best = (sim, match)

        return best[1]

