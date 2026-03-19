import sqlite3
from contextlib import contextmanager

from .config import get_settings


def connect() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def get_conn():
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT UNIQUE NOT NULL,
                customer_query TEXT NOT NULL,
                historical_response TEXT NOT NULL,
                timestamp TEXT,
                tags TEXT
            );

            CREATE TABLE IF NOT EXISTS vector_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL, -- ticket | doc
                source_ref TEXT NOT NULL,  -- ticket_id or doc name
                text TEXT NOT NULL,
                tags TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_vector_items_source_type ON vector_items(source_type);

            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_query TEXT NOT NULL,
                suggestion_text TEXT NOT NULL,
                confidence REAL,
                model_used TEXT,
                response_time_ms INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                status TEXT DEFAULT 'pending' -- pending | accepted | rejected | corrected
            );

            CREATE INDEX IF NOT EXISTS idx_generations_status ON generations(status);

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id INTEGER NOT NULL,
                user_action TEXT NOT NULL, -- accepted | rejected | corrected
                corrected_text TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (generation_id) REFERENCES generations(id) ON DELETE CASCADE
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_generation ON feedback(generation_id);
            """
        )
        conn.commit()


def reset_vector_items() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM vector_items;")
        conn.commit()

