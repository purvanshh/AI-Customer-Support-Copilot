from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from app.embeddings.vector_store import SearchResult, SourceType, VectorStore


@dataclass(frozen=True)
class RetrievedContext:
    """
    Retrieved chunks plus their metadata.
    """

    query: str
    results: list[SearchResult]


class Retriever:
    """
    Retrieves relevant context chunks from the vector store.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        source_types: Optional[Sequence[SourceType]] = None,
    ) -> RetrievedContext:
        """
        Retrieve top-k similar items from the vector store.

        Args:
            query: User query.
            top_k: Number of results to return (after dedupe/filter).
            source_types: Optional filter for ["ticket", "doc"].

        Returns:
            RetrievedContext object.
        """
        results = self.vector_store.similarity_search(
            query,
            top_k=top_k,
            source_types=source_types,
        )
        return RetrievedContext(query=query, results=results)

