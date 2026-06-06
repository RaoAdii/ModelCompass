"""Idempotent migration for Phase 3 analytics columns."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    """Check whether a SQLite column exists.

    Args:
        cursor: SQLite cursor.
        table: Table name.
        column: Column name.

    Returns:
        True when the column exists.
    """
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def run_migration(db_path: str = "app/db/cache.db") -> None:
    """Add Phase 3 analytics columns if missing.

    Args:
        db_path: SQLite database path.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS analytics "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, doc_type VARCHAR NOT NULL, "
            "model_selected VARCHAR NOT NULL, rouge_1 FLOAT NOT NULL, "
            "inference_time_ms FLOAT NOT NULL, was_cached INTEGER NOT NULL, "
            "created_at DATETIME NOT NULL)"
        )
        columns = {
            "semantic_similarity": "REAL DEFAULT NULL",
            "abstractiveness": "REAL DEFAULT NULL",
            "entity_preservation": "REAL DEFAULT NULL",
            "classifier_method": "VARCHAR DEFAULT 'keyword_fallback'",
        }
        for column, definition in columns.items():
            if not _column_exists(cursor, "analytics", column):
                LOGGER.info("Adding analytics column %s", column)
                cursor.execute(f"ALTER TABLE analytics ADD COLUMN {column} {definition}")
        connection.commit()
