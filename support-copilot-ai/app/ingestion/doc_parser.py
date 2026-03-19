from __future__ import annotations

from pathlib import Path

from PyPDF2 import PdfReader


def extract_text(file_path: str | Path) -> str:
    """
    Extract raw text from supported document types.

    Supported formats:
      - .txt
      - .pdf

    Args:
        file_path: Document path.

    Returns:
        Extracted plain text.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suffix = p.suffix.lower()
    if suffix == ".txt":
        return p.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(str(p))
        parts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
        return "\n\n".join(parts)

    raise ValueError(f"Unsupported document format: {suffix}. Use .txt or .pdf")


def chunk_by_paragraphs(text: str, *, min_chunk_chars: int = 1) -> list[str]:
    """
    Chunk text by splitting paragraphs.

    This is intentionally simple for the MVP foundation.

    Args:
        text: Raw extracted text.
        min_chunk_chars: Drop chunks shorter than this.

    Returns:
        List of paragraph chunks.
    """
    normalized = (text or "").strip()
    if not normalized:
        return []

    # Split by one or more blank lines.
    raw_chunks = [c.strip() for c in normalized.split("\n\n") if c.strip()]
    return [c for c in raw_chunks if len(c) >= min_chunk_chars]


def parse_and_chunk(file_path: str | Path) -> list[str]:
    """
    Extract text then chunk it by paragraphs.

    Args:
        file_path: Document path.

    Returns:
        List of paragraph chunks.
    """
    text = extract_text(file_path)
    return chunk_by_paragraphs(text)

