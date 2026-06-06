"""Tests for advanced evaluator metrics."""

from __future__ import annotations

import numpy as np

from app.models.advanced_evaluator import AdvancedEvaluator


class _FakeEmbeddingModel:
    """Small fake embedding model for semantic tests."""

    def encode(self, texts, convert_to_numpy=True):
        """Return deterministic embeddings."""
        return np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]])


class _FakeEntity:
    """Fake spaCy entity."""

    def __init__(self, text: str, label: str) -> None:
        """Store fake entity fields."""
        self.text = text
        self.label_ = label


class _FakeDoc:
    """Fake spaCy doc."""

    def __init__(self, ents) -> None:
        """Store fake entity list."""
        self.ents = ents


def test_abstractiveness_score_bounds() -> None:
    """Abstractiveness score should be a 0..1 float."""
    evaluator = AdvancedEvaluator()
    score = evaluator.abstractiveness_score("the cat sat on the mat", "the dog ran away")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_abstractiveness_identical_text_near_zero() -> None:
    """Identical text should have no novel bigrams."""
    evaluator = AdvancedEvaluator()
    text = "the model summarizes the document"
    assert evaluator.abstractiveness_score(text, text) == 0.0


def test_abstractiveness_different_words_high() -> None:
    """Different summaries should produce higher novelty."""
    evaluator = AdvancedEvaluator()
    score = evaluator.abstractiveness_score("the model summarizes the document", "fresh words appear here")
    assert score > 0.5


def test_semantic_similarity_with_mock(monkeypatch) -> None:
    """Semantic similarity should use the embedding model output."""
    evaluator = AdvancedEvaluator()
    monkeypatch.setattr(evaluator, "_get_embedding_model", lambda: _FakeEmbeddingModel())
    score = evaluator.semantic_similarity("source", "summary")
    assert 0.0 <= score <= 1.0


def test_entity_preservation_with_mock(monkeypatch) -> None:
    """Entity preservation should compare source and summary entities."""
    evaluator = AdvancedEvaluator()

    def fake_nlp(text: str):
        if "OpenAI" in text and "2026" in text:
            return _FakeDoc([_FakeEntity("OpenAI", "ORG"), _FakeEntity("2026", "DATE")])
        return _FakeDoc([_FakeEntity("OpenAI", "ORG")])

    monkeypatch.setattr(evaluator, "_get_nlp", lambda: fake_nlp)
    score = evaluator.entity_preservation_score("OpenAI announced it in 2026", "OpenAI announced it")
    assert 0.0 <= score <= 1.0
    assert score == 0.5


def test_evaluate_all_structure(monkeypatch) -> None:
    """evaluate_all should return metrics for each summary."""
    evaluator = AdvancedEvaluator()
    monkeypatch.setattr(evaluator, "semantic_similarity", lambda source, summary: 0.8)
    monkeypatch.setattr(evaluator, "entity_preservation_score", lambda source, summary: 0.7)
    result = evaluator.evaluate_all("source text", {"summary_bart": "summary text"})
    assert result["summary_bart"]["semantic_similarity"] == 0.8
    assert "abstractiveness" in result["summary_bart"]
    assert result["summary_bart"]["entity_preservation"] == 0.7


def test_metric_failure_returns_none(monkeypatch) -> None:
    """Metric failures should not crash evaluate_all."""
    evaluator = AdvancedEvaluator()

    def fail_metric(source: str, summary: str) -> float:
        raise RuntimeError("boom")

    monkeypatch.setattr(evaluator, "semantic_similarity", fail_metric)
    monkeypatch.setattr(evaluator, "entity_preservation_score", lambda source, summary: 1.0)
    result = evaluator.evaluate_all("source", {"summary_t5": "summary"})
    assert result["summary_t5"]["semantic_similarity"] is None
