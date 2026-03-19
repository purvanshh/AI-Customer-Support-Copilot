from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pydantic import ValidationError

from app.api.schemas import TicketRecord


_REQUIRED_CSV_COLS: dict[str, set[str]] = {
    "id": {"id", "ticket_id", "ticketid", "ticket id"},
    "query": {"query", "customer_query", "customerquery", "customer question", "question", "problem", "issue"},
}
_OPTIONAL_CSV_COLS: dict[str, set[str]] = {
    "response": {"response", "agent_response", "agent response", "answer", "resolution", "reply"},
    "timestamp": {"timestamp", "time", "created_at", "createdat", "date"},
}


def _normalize_columns(columns: Iterable[Any]) -> dict[str, str]:
    """
    Build a mapping from normalized input column name to original column name.

    Args:
        columns: Iterable of CSV headers.

    Returns:
        Mapping: normalized_header -> original_header
    """
    mapping: dict[str, str] = {}
    for c in columns:
        if c is None:
            continue
        s = str(c).strip().lower().replace("-", "_").replace(" ", "_")
        mapping[s] = str(c)
    return mapping


def _find_column(column_map: dict[str, str], candidates: set[str]) -> str | None:
    """
    Find the first matching column from candidate names.

    Args:
        column_map: normalized header -> original header
        candidates: candidate normalized header strings

    Returns:
        Original header string if found, else None.
    """
    for cand in candidates:
        c = cand.strip().lower().replace("-", "_").replace(" ", "_")
        if c in column_map:
            return column_map[c]
    return None


def _rows_to_records(rows: list[dict[str, Any]]) -> list[TicketRecord]:
    """
    Validate and convert raw rows into `TicketRecord` models.

    Args:
        rows: List of dictionaries with keys matching `TicketRecord`.

    Returns:
        List of validated TicketRecord objects.
    """
    records: list[TicketRecord] = []
    errors: list[str] = []
    for i, r in enumerate(rows):
        try:
            records.append(TicketRecord(**r))
        except ValidationError as e:
            errors.append(f"row_index={i} validation_error={str(e)}")
    if errors:
        raise ValueError("Validation errors: " + " | ".join(errors))
    return records


def load_tickets_from_csv(csv_path: str | Path) -> list[TicketRecord]:
    """
    Load and normalize tickets from a CSV file using pandas.

    Required columns (case-insensitive; supports aliases):
      - `id`
      - `query`

    Optional columns:
      - `response`
      - `timestamp`

    Args:
        csv_path: Path to CSV file.

    Returns:
        List of normalized TicketRecord objects.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    column_map = _normalize_columns(df.columns)

    id_col = _find_column(column_map, _REQUIRED_CSV_COLS["id"])
    query_col = _find_column(column_map, _REQUIRED_CSV_COLS["query"])
    response_col = _find_column(column_map, _OPTIONAL_CSV_COLS["response"])
    timestamp_col = _find_column(column_map, _OPTIONAL_CSV_COLS["timestamp"])

    if not id_col or not query_col:
        missing = [name for name, col in [("id", id_col), ("query", query_col)] if not col]
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

    def cell(v: Any) -> Any:
        """Normalize NaN -> None for pydantic."""
        if pd.isna(v):
            return None
        return v

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        response_value = cell(row[response_col]) if response_col else None
        timestamp_value = cell(row[timestamp_col]) if timestamp_col else None
        record: dict[str, Any] = {
            "id": str(cell(row[id_col]) or "").strip(),
            "query": str(cell(row[query_col]) or "").strip(),
            "response": (str(response_value).strip() if response_value not in (None, "") else None),
            "timestamp": (str(timestamp_value).strip() if timestamp_value not in (None, "") else None),
        }
        # Drop fully empty rows.
        if not record["id"] and not record["query"]:
            continue
        rows.append(record)

    valid_rows: list[dict[str, Any]] = []
    for r in rows:
        if not r["id"] or not r["query"]:
            continue
        valid_rows.append(r)

    return _rows_to_records(valid_rows)


def load_tickets_from_json(json_path: str | Path) -> list[TicketRecord]:
    """
    Load and normalize tickets from a JSON file.

    Supported shapes:
      - list of ticket objects
      - { "tickets": [ ... ] }

    Required keys per ticket:
      - `id`
      - `query`

    Optional keys:
      - `response`
      - `timestamp`

    Args:
        json_path: Path to JSON file.

    Returns:
        List of normalized TicketRecord objects.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(str(json_path))

    raw_text = json_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    data = json.loads(raw_text)
    items: Any = data.get("tickets") if isinstance(data, dict) else data
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("JSON must be an array of ticket objects or {\"tickets\": [...]} ")

    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each ticket must be a JSON object")

        def pick(keys: set[str]) -> Any:
            for k in keys:
                if k in item and item[k] not in (None, ""):
                    return item[k]
            return None

        id_value = pick(_REQUIRED_CSV_COLS["id"]) or item.get("id")
        query_value = pick(_REQUIRED_CSV_COLS["query"]) or item.get("query")
        response_value = item.get("response", None)
        timestamp_value = item.get("timestamp", None)

        # Support optional aliases
        if response_value in (None, ""):
            response_value = pick(_OPTIONAL_CSV_COLS["response"])
        if timestamp_value in (None, ""):
            timestamp_value = pick(_OPTIONAL_CSV_COLS["timestamp"])

        rows.append(
            {
                "id": str(id_value or "").strip(),
                "query": str(query_value or "").strip(),
                "response": (str(response_value).strip() if response_value not in (None, "") else None),
                "timestamp": (str(timestamp_value).strip() if timestamp_value not in (None, "") else None),
            }
        )

    valid_rows = [r for r in rows if r["id"] and r["query"]]
    return _rows_to_records(valid_rows)


def load_tickets(file_path: str | Path) -> list[TicketRecord]:
    """
    Load tickets from CSV or JSON based on file extension.

    Args:
        file_path: Path to uploaded tickets file.

    Returns:
        List of normalized TicketRecord objects.
    """
    p = Path(file_path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return load_tickets_from_csv(p)
    if suffix in (".json", ".jsonl"):
        return load_tickets_from_json(p)
    raise ValueError("Unsupported file format for tickets. Use .csv or .json/.jsonl")

