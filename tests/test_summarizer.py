"""Summarizer unit tests."""

from __future__ import annotations

import time

import pytest

from app.exceptions import SummaryTimeoutError
from app.models.summarizer import MultiModelSummarizer


def test_generate_summaries_with_stubbed_models(flask_app, monkeypatch) -> None:
    """Generate all summaries with mocked model calls."""
    summarizer = MultiModelSummarizer()

    def fake_generate(summary_key: str, text: str) -> str:
        return f"{summary_key}: {text[:20]}"

    with flask_app.app_context():
        monkeypatch.setattr(summarizer, "_generate_with_model", fake_generate)
        result = summarizer.generate_summaries("sample source text", timeout_seconds=3)

    assert set(result.keys()) == {"summary_bart", "summary_pegasus", "summary_t5"}
    assert result["summary_bart"].startswith("summary_bart")


def test_generate_summaries_timeout(flask_app, monkeypatch) -> None:
    """Raise timeout error when generation exceeds limit."""
    summarizer = MultiModelSummarizer()

    def slow_generate(summary_key: str, text: str) -> str:
        time.sleep(1.5)
        return "late summary"

    with flask_app.app_context():
        monkeypatch.setattr(summarizer, "_generate_with_model", slow_generate)
        with pytest.raises(SummaryTimeoutError):
            summarizer.generate_summaries("sample source text", timeout_seconds=1)
