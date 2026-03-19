from __future__ import annotations

from pathlib import Path
from typing import Iterable


def load_doc_text(path: str | Path) -> str:
    """
    Supports:
    - .txt, .md
    - .pdf (requires `pypdf` in the environment)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suffix = p.suffix.lower()
    if suffix in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # lazy import
        except Exception as e:
            raise RuntimeError("PDF parsing requires pypdf. Install requirements-optional.txt") from e

        reader = PdfReader(str(p))
        parts: list[str] = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            parts.append(txt)
        return "\n\n".join(parts).strip()

    raise ValueError(f"Unsupported doc type: {suffix}. Use .txt/.md/.pdf")


def chunk_text(text: str, *, chunk_size_chars: int, overlap_chars: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be > 0")

    overlap_chars = max(0, min(overlap_chars, chunk_size_chars - 1))

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap_chars
    return chunks

