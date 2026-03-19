from __future__ import annotations

import os
from pathlib import Path

from dataclasses import dataclass
from functools import lru_cache


def project_root() -> Path:
    """
    Compute the project root directory for `support-copilot-ai/`.

    Returns:
        Path pointing to the `support-copilot-ai` directory.
    """
    # support-copilot-ai/app/core/config.py -> parents[0]=core, [1]=app, [2]=support-copilot-ai
    return Path(__file__).resolve().parents[2]


def data_dirs() -> tuple[Path, Path]:
    """
    Resolve raw and processed data directories.

    Env overrides:
        - DATA_DIR: base directory (defaults to "<project_root>/data")

    Returns:
        (raw_dir, processed_dir)
    """
    base = Path(os.getenv("DATA_DIR", str(project_root() / "data"))).resolve()
    raw_dir = base / "raw"
    processed_dir = base / "processed"
    return raw_dir, processed_dir


@dataclass(frozen=True)
class Settings:
    """
    Runtime configuration for embeddings + retrieval + generation.
    """

    # Embeddings
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "mock")  # mock | sentence-transformers | openai
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    # Vector index persistence
    faiss_index_path: str | None = os.getenv("FAISS_INDEX_PATH")
    faiss_meta_path: str | None = os.getenv("FAISS_META_PATH")

    # Retrieval
    rag_top_k_default: int = int(os.getenv("RAG_TOP_K_DEFAULT", "5"))
    rag_search_k_multiplier: int = int(os.getenv("RAG_SEARCH_K_MULTIPLIER", "5"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "9000"))

    # LLM
    llm_backend: str = os.getenv("LLM_BACKEND", "mock")  # mock | openai | gemini
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    # If true, we attempt to rebuild on mismatched index metadata.
    auto_rebuild_index: bool = os.getenv("AUTO_REBUILD_INDEX", "true").lower() in ("1", "true", "yes")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Resolve Settings once per process.
    """
    raw_dir, processed_dir = data_dirs()
    _ = raw_dir  # keep for symmetry; may be used in the future

    # Default index file locations inside data/processed/
    default_index_path = processed_dir / "faiss.index"
    default_meta_path = processed_dir / "faiss_meta.json"

    s = Settings()
    index_path = s.faiss_index_path or str(default_index_path)
    meta_path = s.faiss_meta_path or str(default_meta_path)

    return Settings(
        embedding_backend=s.embedding_backend,
        embedding_model=s.embedding_model,
        embedding_batch_size=s.embedding_batch_size,
        faiss_index_path=index_path,
        faiss_meta_path=meta_path,
        rag_top_k_default=s.rag_top_k_default,
        rag_search_k_multiplier=s.rag_search_k_multiplier,
        max_context_chars=s.max_context_chars,
        llm_backend=s.llm_backend,
        openai_model=s.openai_model,
        openai_temperature=s.openai_temperature,
        gemini_model=s.gemini_model,
        gemini_temperature=s.gemini_temperature,
        auto_rebuild_index=s.auto_rebuild_index,
    )

