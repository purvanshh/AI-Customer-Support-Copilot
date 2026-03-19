from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Literal

import numpy as np

from app.core.config import get_settings


# ---------------------------------------------------------------------------
# Lightweight vocabulary for the mock TF-IDF embedder.
# Each dimension corresponds to a semantically meaningful keyword bucket.
# This gives mock embeddings directional relevance (not random noise).
# ---------------------------------------------------------------------------

_VOCAB: list[list[str]] = [
    # Billing / Payment
    ["invoice", "billing", "charge", "charged", "payment", "receipt", "plan", "upgrade", "downgrade", "subscription", "price", "pricing", "cost"],
    # Refund / Cancellation
    ["refund", "chargeback", "cancel", "cancellation", "money", "returned", "reimburse", "prorated", "unsubscribe"],
    # Technical / Errors
    ["error", "bug", "crash", "crashing", "broken", "failed", "failure", "timeout", "issue", "fix", "update", "version"],
    # Auth / Login
    ["login", "password", "reset", "credential", "authentication", "2fa", "two-factor", "session", "cookie", "token"],
    # Account / Settings
    ["account", "settings", "profile", "email", "security", "verify", "verification", "configure", "preference"],
    # Data / Import-Export
    ["import", "export", "csv", "json", "xlsx", "data", "upload", "download", "file", "format", "report"],
    # Project / Workspace
    ["project", "workspace", "deleted", "delete", "recover", "recovery", "restore", "trash", "soft-delete"],
    # Support / Help
    ["help", "support", "question", "how", "what", "where", "documentation", "docs", "guide", "tutorial"],
    # API / Integration
    ["api", "endpoint", "request", "rate", "limit", "integration", "webhook", "connect", "connection"],
    # Team / Organization
    ["team", "user", "seat", "admin", "organization", "sso", "member", "collaborate", "permission", "role"],
]

# Flatten for quick lookup: word -> list of bucket indices
_WORD_TO_BUCKETS: dict[str, list[int]] = {}
for _bidx, _bucket in enumerate(_VOCAB):
    for _w in _bucket:
        _WORD_TO_BUCKETS.setdefault(_w, []).append(_bidx)

_NUM_SEMANTIC_DIMS = len(_VOCAB)


class Embedder:
    """
    Returns normalized embeddings so cosine similarity == inner product (for FAISS IndexFlatIP).
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._dim: int | None = None
        self._st_model = None

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim

        # Determine dim lazily.
        if self.settings.embedding_backend == "mock":
            self._dim = 384
            return self._dim

        # For non-mock, we embed a short string to infer dim.
        vec = self.embed_texts(["dim_check"]).astype("float32")
        self._dim = int(vec.shape[1])
        return self._dim

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        # Avoid divide-by-zero; keep deterministic behavior.
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (x / norms).astype("float32")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        backend = self.settings.embedding_backend
        if backend == "mock":
            return self._embed_mock(texts)
        if backend == "sentence_transformers":
            return self._embed_sentence_transformers(texts)
        if backend == "openai":
            return self._embed_openai(texts)
        raise ValueError(f"Unknown embedding_backend: {backend}")

    def _embed_mock(self, texts: list[str]) -> np.ndarray:
        """
        Semantically-aware mock embeddings.

        Strategy:
          dims 0.._NUM_SEMANTIC_DIMS-1  → TF-IDF–like keyword bucket scores
          dims _NUM_SEMANTIC_DIMS..383   → low-magnitude hash-seeded noise
                                           (preserves determinism + slight
                                            differentiation for near-ties)

        This produces vectors where texts containing similar keywords end up
        close together under inner-product / cosine similarity — unlike the
        old SHA256-random approach which was direction-less.
        """
        dim = 384
        vectors = np.zeros((len(texts), dim), dtype="float32")

        for i, text in enumerate(texts):
            # Tokenize: lowercase, split on non-alpha, deduplicate via Counter
            words = re.findall(r"[a-z0-9]+", text.lower())
            word_counts = Counter(words)

            # --- Semantic dimensions (TF-IDF-like) --------------------------
            for word, count in word_counts.items():
                buckets = _WORD_TO_BUCKETS.get(word)
                if buckets:
                    # TF component: log(1 + count) dampens repetition
                    tf = math.log1p(count)
                    for bidx in buckets:
                        vectors[i, bidx] += tf

            # --- Hash-seeded noise for remaining dims ----------------------
            h = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "little", signed=False)
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, 0.05, size=(dim - _NUM_SEMANTIC_DIMS,)).astype("float32")
            vectors[i, _NUM_SEMANTIC_DIMS:] = noise

        return self._normalize(vectors)

    def _embed_sentence_transformers(self, texts: list[str]) -> np.ndarray:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer  # lazy import

            self._st_model = SentenceTransformer(self.settings.embedding_model)
        vectors = self._st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        vectors = vectors.astype("float32", copy=False)
        return self._normalize(vectors)

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        # Note: requires OPENAI_API_KEY in your environment (or via .env).
        from openai import OpenAI  # lazy import

        client = OpenAI()
        resp = client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        vectors = np.array([d.embedding for d in resp.data], dtype="float32")
        return self._normalize(vectors)
