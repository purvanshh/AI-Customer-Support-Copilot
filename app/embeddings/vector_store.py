from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.core.db import connect
from app.embeddings.embedder import Embedder


def _try_import_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:
        return None


class VectorStore:
    """
    Persistent vector search.

    - If FAISS is available: stores `faiss.index` + `faiss_meta.json`.
    - Otherwise: stores `faiss_vectors.npz` + `faiss_ids.npy` (brute force).
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = Embedder()
        self.faiss = _try_import_faiss()

        self._faiss_index = None
        self._bruteforce_vectors: np.ndarray | None = None
        self._bruteforce_ids: np.ndarray | None = None

    def index_meta_path(self) -> str:
        return str(self.settings.faiss_meta_path)

    def vectors_npz_path(self) -> str:
        return os.path.join(self.settings.data_dir, "processed", "faiss_vectors.npz")

    def _read_meta(self) -> dict[str, Any] | None:
        meta_path = Path(self.index_meta_path())
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _write_meta(self, meta: dict[str, Any]) -> None:
        Path(self.settings.faiss_meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def rebuild_from_db(self) -> int:
        """
        Re-embeds all `vector_items` and rebuilds the search index.
        """
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT id, source_type, source_ref, text, tags
                FROM vector_items
                ORDER BY id ASC
                """
            ).fetchall()

        if not rows:
            self._faiss_index = None
            self._bruteforce_vectors = None
            self._bruteforce_ids = None
            # Best-effort: remove old index files.
            if Path(self.settings.faiss_index_path).exists():
                Path(self.settings.faiss_index_path).unlink()
            if Path(self.settings.faiss_meta_path).exists():
                Path(self.settings.faiss_meta_path).unlink()
            npz_path = Path(self.vectors_npz_path())
            if npz_path.exists():
                npz_path.unlink()
            return 0

        item_ids = np.array([int(r["id"]) for r in rows], dtype="int64")
        texts = [str(r["text"]) for r in rows]

        vectors = self.embedder.embed_texts(texts)
        dim = int(vectors.shape[1])
        assert vectors.dtype == np.float32

        meta = {
            "dim": dim,
            "embedding_backend": self.settings.embedding_backend,
            "embedding_model": self.settings.embedding_model,
            "items_count": int(len(rows)),
        }

        if self.faiss is not None:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(dim)
            index = faiss.IndexIDMap(index)
            index.add_with_ids(vectors, item_ids)
            faiss.write_index(index, self.settings.faiss_index_path)
            self._faiss_index = index
            self._bruteforce_vectors = None
            self._bruteforce_ids = None
            self._write_meta(meta)
        else:
            npz_path = self.vectors_npz_path()
            np.savez_compressed(npz_path, vectors=vectors, ids=item_ids)
            self._bruteforce_vectors = vectors
            self._bruteforce_ids = item_ids
            self._faiss_index = None
            self._write_meta(meta | {"backend": "bruteforce"})

        return int(len(rows))

    def _load_faiss_if_needed(self) -> None:
        if self.faiss is None:
            return
        if self._faiss_index is not None:
            return
        if not Path(self.settings.faiss_index_path).exists():
            return
        try:
            import faiss  # type: ignore

            self._faiss_index = faiss.read_index(self.settings.faiss_index_path)
        except Exception:
            self._faiss_index = None

    def _load_bruteforce_if_needed(self) -> None:
        if self.faiss is not None:
            return
        if self._bruteforce_vectors is not None and self._bruteforce_ids is not None:
            return
        npz_path = Path(self.vectors_npz_path())
        if not npz_path.exists():
            return
        data = np.load(npz_path)
        self._bruteforce_vectors = data["vectors"]
        self._bruteforce_ids = data["ids"]

    def is_index_ready(self) -> bool:
        if self.faiss is not None:
            return Path(self.settings.faiss_index_path).exists() and Path(self.settings.faiss_meta_path).exists()
        return Path(self.vectors_npz_path()).exists()

    def search(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        if not self.is_index_ready():
            raise RuntimeError("Vector index not found. Upload tickets first.")

        query_vec = self.embedder.embed_texts([query_text])
        # query_vec is normalized; similarity == dot product.
        if self.faiss is not None:
            self._load_faiss_if_needed()
            if self._faiss_index is None:
                raise RuntimeError("FAISS index could not be loaded.")
            scores, ids = self._faiss_index.search(query_vec, top_k)
            ids = ids[0]
            scores = scores[0]
        else:
            self._load_bruteforce_if_needed()
            if self._bruteforce_vectors is None or self._bruteforce_ids is None:
                raise RuntimeError("Bruteforce vectors could not be loaded.")
            # vectors shape: (n, dim), query_vec: (1, dim)
            sims = (self._bruteforce_vectors @ query_vec[0]).astype("float32")
            top_idx = np.argsort(-sims)[:top_k]
            ids = self._bruteforce_ids[top_idx]
            scores = sims[top_idx]

        # Filter out FAISS placeholder -1 ids.
        results: list[dict[str, Any]] = []
        clean_ids: list[int] = []
        for i in ids:
            ii = int(i)
            if ii >= 0:
                clean_ids.append(ii)
        if not clean_ids:
            return []

        with connect() as conn:
            placeholders = ",".join("?" for _ in clean_ids)
            db_rows = conn.execute(
                f"SELECT id, source_type, source_ref, text, tags FROM vector_items WHERE id IN ({placeholders})",
                clean_ids,
            ).fetchall()

        row_by_id = {int(r["id"]): r for r in db_rows}
        for idx, item_id in enumerate(clean_ids):
            r = row_by_id.get(int(item_id))
            if not r:
                continue
            score = float(scores[idx]) if idx < len(scores) else 0.0
            results.append(
                {
                    "id": int(r["id"]),
                    "source_type": str(r["source_type"]),
                    "source_ref": str(r["source_ref"]),
                    "text": str(r["text"]),
                    "tags": r["tags"],
                    "score": score,
                }
            )
        return results

