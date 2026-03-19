from __future__ import annotations

from typing import Any

from app.embeddings.vector_store import VectorStore


def _parse_tags(tags_value: Any) -> list[str]:
    if tags_value is None:
        return []
    if isinstance(tags_value, list):
        return [str(x).strip() for x in tags_value if str(x).strip()]
    s = str(tags_value).strip()
    if not s:
        return []
    # Stored formats: JSON array string or comma-separated.
    if s.startswith("[") and s.endswith("]"):
        # Best-effort: avoid json import overhead.
        s = s.strip("[]")
        parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
    else:
        parts = s.split(",")
    return [p.strip() for p in parts if p.strip()]


def retrieve_context(customer_query: str, top_k: int, allowed_tags: list[str] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """
    Returns:
      context_text: string to feed the LLM
      sources: list of {source_type, source_ref, score, tags}
    """
    vs = VectorStore()
    results = vs.search(customer_query, top_k=top_k)

    # Optional tag filtering (helps reduce irrelevant context).
    if allowed_tags:
        allowed = {t.strip().lower() for t in allowed_tags if t.strip()}
        filtered: list[dict[str, Any]] = []
        for r in results:
            r_tags = {t.lower() for t in _parse_tags(r.get("tags"))}
            if r_tags & allowed:
                filtered.append(r)
        results = filtered or results  # If everything filtered out, fall back to original results.

    chunks: list[str] = []
    sources: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for r in results:
        source_type = str(r["source_type"])
        source_ref = str(r["source_ref"])
        score = float(r["score"])
        text = str(r["text"])

        # Deduplicate: keep only the highest-scoring result per source_ref.
        dedup_key = f"{source_type}:{source_ref}"
        if dedup_key in seen_refs:
            continue
        seen_refs.add(dedup_key)

        chunks.append(f"[{source_type}:{source_ref} | score={score:.4f}]\n{text}")
        sources.append(
            {
                "source_type": source_type,
                "source_ref": source_ref,
                "tags": _parse_tags(r.get("tags")),
                "score": score,
            }
        )

    context_text = "\n\n".join(chunks)
    return context_text, sources

