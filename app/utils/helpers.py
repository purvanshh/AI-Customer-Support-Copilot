from __future__ import annotations

import datetime as dt
from typing import Any


def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if str(t).strip()]
    s = str(tags).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        # naive JSON-ish comma split
        s = s.strip("[]")
        parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
        return [p for p in parts if p]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]

