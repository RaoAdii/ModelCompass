"""Tests for SQLite document cache utility."""

from __future__ import annotations

from pathlib import Path

from app.utils.cache import DocumentCache, compute_doc_hash


def test_cache_set_get_exists(tmp_path: Path) -> None:
    """Cache should store and retrieve summary payloads."""
    db_path = tmp_path / "cache.db"
    cache = DocumentCache(db_path=str(db_path))
    text = "This is a repeatable document body."
    doc_hash = compute_doc_hash(text)
    summaries = {"summary_bart": "a", "summary_pegasus": "b", "summary_t5": "c"}
    rouge_scores = {"pairwise_rouge": {"a_vs_b": {"rouge1": 0.25}}}

    assert cache.exists(doc_hash) is False
    assert cache.set(doc_hash, summaries, rouge_scores) is True
    assert cache.exists(doc_hash) is True

    payload = cache.get(doc_hash)
    assert payload is not None
    assert payload["summaries"] == summaries
    assert payload["rouge_scores"] == rouge_scores


def test_cache_hit_count_increment(tmp_path: Path) -> None:
    """Cache hit count should increment on demand."""
    db_path = tmp_path / "cache.db"
    cache = DocumentCache(db_path=str(db_path))
    doc_hash = compute_doc_hash("content")
    cache.set(doc_hash, {"summary_bart": "cached"}, {"pairwise_rouge": {}})

    cache.increment_hit_count(doc_hash)
    cache.increment_hit_count(doc_hash)
    payload = cache.get(doc_hash)
    assert payload is not None
    assert payload["hit_count"] == 2


def test_cache_clear_specific_and_all(tmp_path: Path) -> None:
    """Cache clear should remove one key or all keys."""
    db_path = tmp_path / "cache.db"
    cache = DocumentCache(db_path=str(db_path))
    h1 = compute_doc_hash("doc1")
    h2 = compute_doc_hash("doc2")
    cache.set(h1, {"summary_bart": "x"}, {"pairwise_rouge": {}})
    cache.set(h2, {"summary_bart": "y"}, {"pairwise_rouge": {}})

    assert cache.clear(doc_hash=h1) == 1
    assert cache.exists(h1) is False
    assert cache.exists(h2) is True

    assert cache.clear() == 1
    assert cache.exists(h2) is False


def test_cache_stats_shape(tmp_path: Path) -> None:
    """Cache stats should include expected keys and numeric values."""
    db_path = tmp_path / "cache.db"
    cache = DocumentCache(db_path=str(db_path))
    h1 = compute_doc_hash("doc-stat-1")
    cache.set(h1, {"summary_bart": "z"}, {"pairwise_rouge": {}})
    cache.increment_hit_count(h1)

    stats = cache.get_stats()
    expected_keys = {
        "total_entries",
        "total_hits",
        "hit_rate",
        "ttl_days",
        "stale_entries",
        "oldest_entry_ts",
        "db_size_bytes",
    }
    assert expected_keys.issubset(stats.keys())
    assert stats["total_entries"] >= 1
    assert stats["total_hits"] >= 1
    assert 0.0 <= stats["hit_rate"] <= 1.0
