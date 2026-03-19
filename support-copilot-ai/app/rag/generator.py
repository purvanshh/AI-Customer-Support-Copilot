from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from app.core.config import get_settings
from app.embeddings.vector_store import SearchResult
from app.services.llm_service import LLMService


@dataclass(frozen=True)
class GenerationResult:
    """
    Final model output plus grounding metadata.
    """

    response: str
    sources: list[SearchResult]
    confidence: float


class Generator:
    """
    Build prompts from retrieved context and generate an answer via LLM.
    """

    def __init__(self, llm: LLMService) -> None:
        self.llm = llm
        self.settings = get_settings()

    def _truncate_context(self, results: Sequence[SearchResult]) -> list[SearchResult]:
        """
        Limit context size to prevent overly large prompts.
        """
        budget = self.settings.max_context_chars
        used = 0
        kept: list[SearchResult] = []
        for r in results:
            text = r.text.strip()
            add = len(text) + 200  # small overhead per chunk
            if used + add > budget and kept:
                break
            if add > budget and not kept:
                # If a single chunk is huge, truncate it to fit.
                r = SearchResult(
                    source_type=r.source_type,
                    source_ref=r.source_ref,
                    text=text[: max(0, budget - 200)],
                    score=r.score,
                )
            kept.append(r)
            used += len(r.text) + 200
        return kept

    def build_prompt(self, *, query: str, context_results: Sequence[SearchResult]) -> str:
        """
        Create a grounded prompt template.
        """
        system = (
            "You are SupportCopilot AI, a customer support assistant. "
            "Write a helpful, concise answer using ONLY the provided context. "
            "If context is insufficient, ask 1-2 clarifying questions. "
            "Do not invent policy details."
        )

        context_results = self._truncate_context(context_results)
        context_blocks: list[str] = []
        for i, r in enumerate(context_results, start=1):
            context_blocks.append(
                f"[CONTEXT {i}] source={r.source_type}:{r.source_ref}\n{r.text.strip()}"
            )
        context = "\n\n".join(context_blocks) if context_blocks else "(No context retrieved.)"

        user = f"Here is the customer query:\n{query}\n\nRelevant context:\n{context}\n\nNow write the response message body."
        return f"{system}\n\n{user}"

    def _confidence_from_results(self, results: Sequence[SearchResult]) -> float:
        """
        Estimate confidence from similarity scores.
        """
        if not results:
            return 0.0
        scores = [max(-1.0, min(1.0, r.score)) for r in results]
        # Convert dot-product in [-1,1] to [0,1].
        avg = sum(scores) / len(scores)
        return float((avg + 1.0) / 2.0)

    def generate(
        self,
        *,
        query: str,
        retrieved_results: Sequence[SearchResult],
    ) -> GenerationResult:
        """
        Generate a grounded response.
        """
        # Deduplication is already handled in vector_store, but we keep a final safety pass.
        seen = set()
        deduped: list[SearchResult] = []
        for r in retrieved_results:
            key = r.source_ref
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)

        prompt = self.build_prompt(query=query, context_results=deduped)
        response = self.llm.generate_response(prompt)
        confidence = self._confidence_from_results(deduped)
        return GenerationResult(response=response, sources=list(deduped), confidence=confidence)

