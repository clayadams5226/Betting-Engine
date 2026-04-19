"""SQLite caching layer to avoid re-pulling same-day stats."""

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "cache.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def get(key: str) -> dict | None:
    """Get a cached value by key. Returns None if not found."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None
    finally:
        conn.close()


def put(key: str, value: dict) -> None:
    """Store a value in the cache."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, default=str), datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_daily(namespace: str, game_date: date) -> dict | None:
    """Get a cached value for a specific date."""
    key = f"{namespace}:{game_date.isoformat()}"
    return get(key)


def put_daily(namespace: str, game_date: date, value: dict) -> None:
    """Store a value for a specific date."""
    key = f"{namespace}:{game_date.isoformat()}"
    put(key, value)


def clear_old(days: int = 7) -> int:
    """Delete cache entries older than N days. Returns count deleted."""
    conn = _get_conn()
    try:
        cutoff = datetime.now().isoformat()
        cursor = conn.execute(
            "DELETE FROM cache WHERE created_at < date(?, ?)",
            (cutoff, f"-{days} days"),
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def clear_all() -> None:
    """Clear all cached data."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM cache")
        conn.commit()
    finally:
        conn.close()
