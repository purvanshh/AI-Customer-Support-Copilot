from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from app.core.config import get_settings
from app.embeddings.embedder import Embedder


SourceType = Literal["ticket", "doc"]


@dataclass
class VectorDocument:
    """
    A unit of text to embed and store in vector DB.
    """

    source_type: SourceType
    source_ref: str
    text: str


@dataclass
class SearchResult:
    """
    A single similarity search result.
    """

    source_type: SourceType
    source_ref: str
    text: str
    score: float


def _try_import_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:
        return None


class VectorStore:
    """
    Local persistent vector store backed by FAISS (with an offline fallback).
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = Embedder()
        self.faiss = _try_import_faiss()

        self._index = None
        self._dim: Optional[int] = None
        self._items_by_id: Dict[int, VectorDocument] = {}

    @property
    def index_path(self) -> Path:
        return Path(self.settings.faiss_index_path or (self.settings.faiss_index_path))

    @property
    def meta_path(self) -> Path:
        return Path(self.settings.faiss_meta_path or (self.settings.faiss_meta_path))

    def is_index_ready(self) -> bool:
        """
        Check whether index + metadata exist.
        """
        return self.index_path.exists() and self.meta_path.exists()

    def _write_meta(self, meta: dict[str, Any]) -> None:
        """
        Persist metadata to disk.
        """
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _load_meta(self) -> dict[str, Any]:
        """
        Load metadata from disk.
        """
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def load(self) -> None:
        """
        Load FAISS index and metadata into memory.
        """
        if not self.is_index_ready():
            raise FileNotFoundError("FAISS index not found. Build the index first.")

        if self.faiss is None:
            raise RuntimeError("FAISS is not available in this environment.")

        import faiss  # type: ignore

        meta = self._load_meta()
        dim = int(meta["dim"])
        self._dim = dim
        items = meta["items"]
        self._items_by_id = {}
        for it in items:
            doc = VectorDocument(source_type=it["source_type"], source_ref=it["source_ref"], text=it["text"])
            self._items_by_id[int(it["id"])] = doc

        self._index = faiss.read_index(str(self.index_path))

    def reset(self) -> None:
        """
        Reset in-memory state only.
        """
        self._index = None
        self._dim = None
        self._items_by_id = {}

    def build_index(self, documents: Sequence[VectorDocument], *, built_from: dict[str, Any]) -> None:
        """
        Build a new FAISS index from documents and persist it.

        Args:
            documents: VectorDocument objects to embed/store.
            built_from: Source file manifest for staleness checks/debug.
        """
        if not documents:
            raise ValueError("No documents provided for vector index build.")

        if self.faiss is None:
            raise RuntimeError("FAISS is not available in this environment.")

        import faiss  # type: ignore

        texts = [d.text for d in documents]
        vectors = self.embedder.embed_texts(texts, batch_size=self.settings.embedding_batch_size)
        vectors = vectors.astype("float32")
        dim = int(vectors.shape[1])
        self._dim = dim

        index = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(index)

        ids = np.arange(len(documents), dtype="int64")
        index.add_with_ids(vectors, ids)
        self._index = index

        items = []
        self._items_by_id = {}
        for i, d in enumerate(documents):
            doc_id = int(ids[i])
            items.append(
                {
                    "id": doc_id,
                    "source_type": d.source_type,
                    "source_ref": d.source_ref,
                    "text": d.text,
                }
            )
            self._items_by_id[doc_id] = d

        meta = {
            "dim": dim,
            "items": items,
            "built_from": built_from,
        }

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        self._write_meta(meta)

    def similarity_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_types: Optional[Sequence[SourceType]] = None,
        search_k: Optional[int] = None,
        dedupe_key: Optional[str] = "source_ref",
    ) -> List[SearchResult]:
        """
        Run similarity search and return top-k results.

        Args:
            query: User query text.
            top_k: Number of results after filtering/deduping.
            source_types: Optional filter to ticket/doc.
            search_k: Raw FAISS search K (before filtering).
            dedupe_key: Field to deduplicate by ("source_ref" is recommended).
        """
        if self._index is None:
            self.load()

        if self._dim is None:
            raise RuntimeError("Vector index dim not loaded.")

        if self.faiss is None:
            raise RuntimeError("FAISS is not available.")

        search_k_final = search_k or int(top_k * self.settings.rag_search_k_multiplier)
        search_k_final = max(search_k_final, top_k)

        query_vec = self.embedder.embed_texts([query], batch_size=1)
        query_vec = query_vec.astype("float32")
        scores, ids = self._index.search(query_vec, search_k_final)

        scores_row = scores[0]
        ids_row = ids[0]

        filter_set = set(source_types) if source_types else None
        seen: set[str] = set()

        results: List[SearchResult] = []
        for score, doc_id in zip(scores_row, ids_row):
            doc_id_int = int(doc_id)
            if doc_id_int < 0:
                continue
            doc = self._items_by_id.get(doc_id_int)
            if not doc:
                continue
            if filter_set is not None and doc.source_type not in filter_set:
                continue

            key_val = doc.source_ref if dedupe_key == "source_ref" else doc.source_type
            if key_val in seen:
                continue
            seen.add(key_val)

            results.append(SearchResult(source_type=doc.source_type, source_ref=doc.source_ref, text=doc.text, score=float(score)))
            if len(results) >= top_k:
                break

        return results

