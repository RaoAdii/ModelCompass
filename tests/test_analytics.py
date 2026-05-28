"""Tests for v2 analytics and cache-aware summarize endpoint."""

from __future__ import annotations

import io
from pathlib import Path


def _mock_generate_summaries(text: str, timeout_seconds: int, use_parallel: bool = False):
    return {
        "summary_bart": f"bart::{text[:20]}",
        "summary_pegasus": f"pegasus::{text[:20]}",
        "summary_t5": f"t5::{text[:20]}",
    }


def _mock_evaluate_summaries(source_text: str, summaries):
    return {
        "pairwise_rouge": {
            "summary_bart_vs_summary_pegasus": {"rouge1": 0.3, "rouge2": 0.2, "rougeL": 0.25},
            "summary_bart_vs_summary_t5": {"rouge1": 0.28, "rouge2": 0.18, "rougeL": 0.22},
            "summary_pegasus_vs_summary_t5": {"rouge1": 0.26, "rouge2": 0.16, "rougeL": 0.21},
        },
        "custom_metrics": {
            "summary_bart": {"word_count": 10.0, "compression_ratio": 0.2},
            "summary_pegasus": {"word_count": 11.0, "compression_ratio": 0.22},
            "summary_t5": {"word_count": 9.0, "compression_ratio": 0.18},
        },
    }


def test_v2_summarize_cache_miss_then_hit(client, monkeypatch, tmp_path: Path) -> None:
    """Second identical request should be served from cache."""
    client.application.config["DB_PATH"] = str(tmp_path / "analytics_cache.db")
    monkeypatch.setattr("app.routes.api_v2.summarizer.generate_summaries", _mock_generate_summaries)
    monkeypatch.setattr("app.routes.api_v2.evaluator.evaluate_summaries", _mock_evaluate_summaries)

    payload = {
        "file": (io.BytesIO(b"Abstract methodology results conclusion references"), "paper.txt"),
    }
    response_first = client.post("/api/v2/summarize", data=payload, content_type="multipart/form-data")
    assert response_first.status_code == 200
    body_first = response_first.get_json()
    assert body_first["metadata"]["was_cached"] is False
    assert body_first["metadata"]["routed_model"] in {"bart", "pegasus", "t5"}

    payload_repeat = {
        "file": (io.BytesIO(b"Abstract methodology results conclusion references"), "paper.txt"),
    }
    response_second = client.post("/api/v2/summarize", data=payload_repeat, content_type="multipart/form-data")
    assert response_second.status_code == 200
    body_second = response_second.get_json()
    assert body_second["metadata"]["was_cached"] is True
    assert body_second["summaries"] == body_first["summaries"]
    assert body_second["rouge_scores"] == body_first["rouge_scores"]


def test_v2_analytics_endpoint_structure(client, monkeypatch, tmp_path: Path) -> None:
    """Analytics endpoint should return summary and grouped metrics."""
    client.application.config["DB_PATH"] = str(tmp_path / "analytics_stats.db")
    monkeypatch.setattr("app.routes.api_v2.summarizer.generate_summaries", _mock_generate_summaries)
    monkeypatch.setattr("app.routes.api_v2.evaluator.evaluate_summaries", _mock_evaluate_summaries)

    data = {"file": (io.BytesIO(b"reported according to agency witness"), "news.txt")}
    summarize_response = client.post("/api/v2/summarize", data=data, content_type="multipart/form-data")
    assert summarize_response.status_code == 200

    analytics_response = client.get("/api/v2/analytics")
    assert analytics_response.status_code == 200
    analytics_body = analytics_response.get_json()
    assert "summary" in analytics_body
    assert "by_doc_type" in analytics_body
    assert "by_model" in analytics_body
    assert "recent_summaries" in analytics_body

    assert analytics_body["summary"]["total_documents"] >= 1
    assert isinstance(analytics_body["recent_summaries"], list)


def test_v2_cache_clear_endpoint(client, monkeypatch, tmp_path: Path) -> None:
    """Cache clear endpoint should remove entries."""
    client.application.config["DB_PATH"] = str(tmp_path / "analytics_clear.db")
    monkeypatch.setattr("app.routes.api_v2.summarizer.generate_summaries", _mock_generate_summaries)
    monkeypatch.setattr("app.routes.api_v2.evaluator.evaluate_summaries", _mock_evaluate_summaries)

    data = {"file": (io.BytesIO(b"official launch effective immediately"), "announcement.txt")}
    first = client.post("/api/v2/summarize", data=data, content_type="multipart/form-data")
    assert first.status_code == 200
    doc_hash = first.get_json()["metadata"]["doc_hash"]

    clear_one = client.post("/api/v2/cache/clear", json={"doc_hash": doc_hash})
    assert clear_one.status_code == 200
    assert clear_one.get_json()["cleared"] == 1

    clear_all = client.post("/api/v2/cache/clear", json={})
    assert clear_all.status_code == 200
    assert clear_all.get_json()["cleared"] >= 0
