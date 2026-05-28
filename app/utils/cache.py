"""SQLite-backed caching layer for summarized documents."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.schema import CacheEntry, create_session_factory, init_db


def compute_doc_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash for document text.

    Args:
        text: Normalized document text.

    Returns:
        SHA-256 hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentCache:
    """Simple SQLite-based cache for summaries."""

    def __init__(self, db_path: str = "app/db/cache.db") -> None:
        """Initialize cache and ensure DB schema exists.

        Args:
            db_path: SQLite database file path.
        """
        self.db_path = db_path
        init_db(db_path=db_path)
        self._session_factory = create_session_factory(db_path=db_path)

    def _session(self) -> Session:
        """Create a new SQLAlchemy session.

        Returns:
            Active SQLAlchemy session.
        """
        return self._session_factory()

    def get(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached summaries by hash.

        Args:
            doc_hash: SHA-256 document hash.

        Returns:
            Cache payload or None when not found.
        """
        with self._session() as session:
            entry = session.get(CacheEntry, doc_hash)
            if entry is None:
                return None
            return {
                "doc_hash": entry.doc_hash,
                "summaries": json.loads(entry.summaries_json),
                "rouge_scores": json.loads(entry.rouge_scores_json),
                "hit_count": entry.hit_count,
                "created_at": entry.created_at.isoformat(),
            }

    def set(
        self,
        doc_hash: str,
        summaries: Dict[str, Any],
        rouge_scores: Dict[str, Any],
    ) -> bool:
        """Store summaries and scores in cache.

        Args:
            doc_hash: SHA-256 document hash.
            summaries: Summaries payload.
            rouge_scores: ROUGE evaluation payload.

        Returns:
            True when entry was persisted successfully.
        """
        created_at = datetime.now(timezone.utc)
        with self._session() as session:
            entry = session.get(CacheEntry, doc_hash)
            if entry is None:
                entry = CacheEntry(
                    doc_hash=doc_hash,
                    summaries_json=json.dumps(summaries),
                    rouge_scores_json=json.dumps(rouge_scores),
                    hit_count=0,
                    created_at=created_at,
                )
                session.add(entry)
            else:
                entry.summaries_json = json.dumps(summaries)
                entry.rouge_scores_json = json.dumps(rouge_scores)
            session.commit()
        return True

    def exists(self, doc_hash: str) -> bool:
        """Check if document hash exists in cache.

        Args:
            doc_hash: SHA-256 document hash.

        Returns:
            True if cache entry exists.
        """
        with self._session() as session:
            query = select(CacheEntry.doc_hash).where(CacheEntry.doc_hash == doc_hash)
            return session.execute(query).first() is not None

    def clear(self, doc_hash: Optional[str] = None) -> int:
        """Clear specific or all cache entries.

        Args:
            doc_hash: Optional document hash. If omitted, all entries are removed.

        Returns:
            Number of deleted rows.
        """
        with self._session() as session:
            if doc_hash:
                entry = session.get(CacheEntry, doc_hash)
                if entry is None:
                    return 0
                session.delete(entry)
                session.commit()
                return 1

            deleted = session.query(CacheEntry).delete()
            session.commit()
            return int(deleted)

    def increment_hit_count(self, doc_hash: str) -> None:
        """Increment cache hit counter for an entry.

        Args:
            doc_hash: SHA-256 document hash.
        """
        with self._session() as session:
            entry = session.get(CacheEntry, doc_hash)
            if entry is None:
                return
            entry.hit_count += 1
            session.commit()

    def get_stats(self) -> Dict[str, float]:
        """Return cache statistics.

        TTL policy is informational only (30 days) and not enforced automatically.

        Returns:
            Stats dictionary with totals and estimated hit rate.
        """
        with self._session() as session:
            total_entries = session.query(func.count(CacheEntry.doc_hash)).scalar() or 0
            total_hits = session.query(func.sum(CacheEntry.hit_count)).scalar() or 0
            earliest = session.query(func.min(CacheEntry.created_at)).scalar()

        ttl_days = 30
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        stale_entries = 0
        with self._session() as session:
            stale_entries = (
                session.query(func.count(CacheEntry.doc_hash))
                .filter(CacheEntry.created_at < cutoff)
                .scalar()
                or 0
            )

        denominator = total_entries + total_hits
        hit_rate = float(total_hits / denominator) if denominator else 0.0
        db_size_bytes = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0

        return {
            "total_entries": float(total_entries),
            "total_hits": float(total_hits),
            "hit_rate": round(hit_rate, 4),
            "ttl_days": float(ttl_days),
            "stale_entries": float(stale_entries),
            "oldest_entry_ts": earliest.isoformat() if earliest else "",
            "db_size_bytes": float(db_size_bytes),
        }

