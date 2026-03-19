from __future__ import annotations

from app.models.classifier import ClassificationResult, classify_keywords


class ClassificationService:
    """
    Ticket/query classification service.
    """

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify an incoming ticket.

        Args:
            query: Customer query text.

        Returns:
            ClassificationResult
        """
        return classify_keywords(query)

