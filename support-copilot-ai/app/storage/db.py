from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.core.config import data_dirs, get_settings


def db_path() -> Path:
    """
    Compute SQLite DB path for feedback storage.

    Uses:
      - DATA_DIR=<base> (from app.core.config)
    """
    _, processed_dir = data_dirs()
    return processed_dir / "supportcopilot_feedback.db"


def _connect() -> sqlite3.Connection:
    """
    Create a new SQLite connection.

    Returns:
        sqlite3.Connection with row access by column name.
    """
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """
    Provide a transactional connection scoped to the caller.
    """
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """
    Initialize database schema if it doesn't exist.
    """
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                corrected_response TEXT,
                rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
            CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query);
            """
        )
        conn.commit()

