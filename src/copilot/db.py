"""SQLite init + helpers for escalations, interactions, feedback."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import SQLITE_PATH


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    intent TEXT,
    severity_label TEXT,
    severity_score REAL,
    reason_terms TEXT,
    neighbors_json TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    resolution TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT,
    resolved_by TEXT
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    route TEXT NOT NULL,
    intent TEXT,
    confidence REAL,
    severity_label TEXT,
    severity_score REAL,
    answer TEXT,
    escalation_id INTEGER,
    latency_ms INTEGER,
    thumbs TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    thumbs TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (interaction_id) REFERENCES interactions(id)
);
"""


def ensure_db(path: Path | None = None) -> Path:
    db_path = path or SQLITE_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    return db_path


@contextmanager
def get_conn(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    db_path = ensure_db(path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def execute(sql: str, params: tuple[Any, ...] = (), path: Path | None = None) -> int:
    with get_conn(path) as conn:
        cur = conn.execute(sql, params)
        return int(cur.lastrowid or 0)


def fetchall(sql: str, params: tuple[Any, ...] = (), path: Path | None = None) -> list[dict]:
    with get_conn(path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def fetchone(sql: str, params: tuple[Any, ...] = (), path: Path | None = None) -> dict | None:
    with get_conn(path) as conn:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None
