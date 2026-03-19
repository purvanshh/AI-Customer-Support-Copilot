from __future__ import annotations

import hashlib
import math
from typing import Literal

import numpy as np

from app.core.config import get_settings


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
        dim = 384
        vectors = np.zeros((len(texts), dim), dtype="float32")
        for i, t in enumerate(texts):
            # Deterministic pseudo-random vector from text hash.
            h = hashlib.sha256(t.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "little", signed=False)
            rng = np.random.default_rng(seed)
            v = rng.normal(0, 1, size=(dim,)).astype("float32")
            vectors[i] = v
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

