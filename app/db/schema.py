"""SQLAlchemy schema for cache and analytics tables."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, sessionmaker

Base = declarative_base()


class CacheEntry(Base):
    """Cached summary payload keyed by document hash."""

    __tablename__ = "cache"

    doc_hash: Mapped[str] = mapped_column(String, primary_key=True)
    summaries_json: Mapped[str] = mapped_column(String, nullable=False)
    rouge_scores_json: Mapped[str] = mapped_column(String, nullable=False)
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class AnalyticsLog(Base):
    """Per-request analytics logs for v2 summarization calls."""

    __tablename__ = "analytics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_type: Mapped[str] = mapped_column(String, nullable=False)
    model_selected: Mapped[str] = mapped_column(String, nullable=False)
    rouge_1: Mapped[float] = mapped_column(Float, nullable=False)
    inference_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    was_cached: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    semantic_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    abstractiveness: Mapped[float | None] = mapped_column(Float, nullable=True)
    entity_preservation: Mapped[float | None] = mapped_column(Float, nullable=True)
    classifier_method: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="keyword_fallback",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


def _ensure_parent_dir(db_path: str) -> None:
    """Create parent folder for SQLite file if needed.

    Args:
        db_path: SQLite database file path.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def create_db_engine(db_path: str = "app/db/cache.db"):
    """Create SQLAlchemy engine for SQLite database.

    Args:
        db_path: SQLite database file path.

    Returns:
        SQLAlchemy engine instance.
    """
    _ensure_parent_dir(db_path)
    return create_engine(f"sqlite:///{db_path}", future=True)


def create_session_factory(db_path: str = "app/db/cache.db") -> sessionmaker:
    """Create session factory bound to SQLite engine.

    Args:
        db_path: SQLite database file path.

    Returns:
        SQLAlchemy sessionmaker.
    """
    engine = create_db_engine(db_path=db_path)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db(db_path: str = "app/db/cache.db") -> None:
    """Initialize database schema.

    Args:
        db_path: SQLite database file path.
    """
    engine = create_db_engine(db_path=db_path)
    Base.metadata.create_all(engine)
