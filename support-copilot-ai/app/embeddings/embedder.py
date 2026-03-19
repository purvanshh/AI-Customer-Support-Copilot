from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np

from app.core.config import get_settings


EmbeddingBackend = Literal["mock", "sentence-transformers", "openai"]


@dataclass
class EmbedResult:
    """
    Embedding results for a list of texts.
    """

    vectors: np.ndarray  # shape: (n, dim), dtype float32


class Embedder:
    """
    Embedding pipeline with:
      - batching
      - deterministic `mock` backend (for offline/dev)
      - optional OpenAI embeddings or sentence-transformers

    Note:
        Sentence-transformers and OpenAI are loaded lazily to avoid import-time cost.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._dim: Optional[int] = None
        self._st_model = None
        self._cache: Dict[str, np.ndarray] = {}

    def _hash_text(self, text: str) -> str:
        """
        Create a stable key for caching.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize so cosine similarity == inner product for normalized vectors.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vectors / norms).astype("float32")

    def _embed_mock(self, texts: Sequence[str]) -> np.ndarray:
        """
        Deterministic pseudo-embeddings (unit vectors).
        """
        dim = self._dim or 384
        self._dim = dim

        vectors = np.zeros((len(texts), dim), dtype="float32")
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "little", signed=False)
            rng = np.random.default_rng(seed)
            v = rng.normal(0, 1, size=(dim,)).astype("float32")
            vectors[i] = v

        return self._normalize(vectors)

    def _embed_sentence_transformers(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed using sentence-transformers.
        """
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer  # lazy import

            self._st_model = SentenceTransformer(self.settings.embedding_model)
        vectors = self._st_model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        vectors = np.asarray(vectors, dtype="float32")
        if self._dim is None:
            self._dim = int(vectors.shape[1])
        return self._normalize(vectors)

    def _embed_openai(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed using OpenAI embeddings API.
        """
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        from openai import OpenAI  # lazy import

        client = OpenAI(api_key=api_key)
        # `embedding_model` can be either an embeddings model name or a sentence-transformers id
        resp = client.embeddings.create(model=self.settings.embedding_model, input=list(texts))
        vectors = np.asarray([d.embedding for d in resp.data], dtype="float32")
        if self._dim is None:
            self._dim = int(vectors.shape[1])
        return self._normalize(vectors)

    def embed_texts(self, texts: List[str], *, batch_size: Optional[int] = None, use_cache: bool = True) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: Input texts.
            batch_size: Optional override for batching.
            use_cache: Enable embedding caching by text hash.

        Returns:
            Vectors array in the same order as input.
        """
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        backend: EmbeddingBackend = self.settings.embedding_backend  # type: ignore[assignment]
        bs = batch_size or self.settings.embedding_batch_size

        # Cache lookup
        keys = [self._hash_text(t) for t in texts]
        if use_cache:
            cached = [self._cache.get(k) for k in keys]
        else:
            cached = [None] * len(texts)

        missing_indices = [i for i, v in enumerate(cached) if v is None]
        if missing_indices:
            # Embed in batches, preserve ordering
            to_embed = [texts[i] for i in missing_indices]
            vectors_out: List[np.ndarray] = []
            for start in range(0, len(to_embed), bs):
                chunk = to_embed[start : start + bs]
                if backend == "mock":
                    vectors_out.append(self._embed_mock(chunk))
                elif backend == "sentence-transformers":
                    vectors_out.append(self._embed_sentence_transformers(chunk))
                elif backend == "openai":
                    vectors_out.append(self._embed_openai(chunk))
                else:
                    raise ValueError(f"Unknown embedding_backend: {backend}")
            embedded = np.vstack(vectors_out)

            for idx, i in enumerate(missing_indices):
                vec = embedded[idx]
                self._cache[keys[i]] = vec
                cached[i] = vec

        # At this point cached should be fully populated.
        vectors = np.stack([v for v in cached if v is not None], axis=0).astype("float32")
        if self._dim is None and vectors.shape[1] > 0:
            self._dim = int(vectors.shape[1])
        return vectors

