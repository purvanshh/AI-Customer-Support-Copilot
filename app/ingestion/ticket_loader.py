from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class LoadedTicket:
    ticket_id: str
    customer_query: str
    response: str
    timestamp: str | None = None
    tags: list[str] | None = None


def _norm_header(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def _pick_first(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _parse_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    # Common formats: "a,b,c" or "a|b|c"
    if "," in s:
        parts = s.split(",")
    elif "|" in s:
        parts = s.split("|")
    else:
        parts = [s]
    return [p.strip() for p in parts if p.strip()]


def _map_ticket_row(row: dict[str, Any]) -> LoadedTicket:
    # csv.DictReader can emit extra columns under the key `None` when a row has
    # more fields than the header (e.g., unquoted commas inside a field).
    normalized: dict[str, Any] = {}
    for k, v in row.items():
        if k is None:
            continue
        normalized[_norm_header(str(k))] = v

    ticket_id = _pick_first(normalized, ["ticket_id", "ticketid", "ticket", "id"])
    customer_query = _pick_first(
        normalized,
        ["customer_query", "customerquery", "customer_question", "query", "question", "problem", "issue"],
    )
    response = _pick_first(normalized, ["response", "agent_response", "answer", "resolution", "reply"])
    timestamp = _pick_first(normalized, ["timestamp", "time", "created_at", "createdat", "date"])
    tags = _parse_tags(_pick_first(normalized, ["tags", "tag", "label", "labels"]))

    if ticket_id is None or customer_query is None or response is None:
        missing = [k for k, v in [("ticket_id", ticket_id), ("customer_query", customer_query), ("response", response)] if v is None]
        raise ValueError(f"Missing required ticket fields: {', '.join(missing)}")

    return LoadedTicket(
        ticket_id=str(ticket_id),
        customer_query=str(customer_query),
        response=str(response),
        timestamp=str(timestamp) if timestamp not in (None, "") else None,
        tags=tags,
    )


def load_tickets(path: str | Path) -> list[LoadedTicket]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix in (".json", ".jsonl"):
        return _load_json(path)
    raise ValueError(f"Unsupported tickets file type: {suffix}. Use .csv or .json/.jsonl")


def _load_csv(path: Path) -> list[LoadedTicket]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing headers")
        tickets: list[LoadedTicket] = []
        for row in reader:
            tickets.append(_map_ticket_row(row))
        return tickets


def _load_json(path: Path) -> list[LoadedTicket]:
    # Supports:
    # - JSON array of ticket objects
    # - JSON object with {"tickets": [...]}.
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    if path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        items = data.get("tickets") if isinstance(data, dict) else data
        if items is None:
            raise ValueError("JSON must be an array or an object with key 'tickets'")

    if not isinstance(items, list):
        raise ValueError("JSON tickets must be a list")

    tickets: list[LoadedTicket] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each ticket must be a JSON object")
        tickets.append(_map_ticket_row(item))
    return tickets

