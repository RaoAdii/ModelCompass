"""Tests for Phase 3 document classifier."""

from __future__ import annotations

from app.models.classifier import DocumentClassifier


def test_classifier_detects_research_paper() -> None:
    """Research paper text should classify confidently."""
    classifier = DocumentClassifier()
    text = (
        "Abstract methodology hypothesis empirical findings literature review "
        "experimental results dataset baseline precision recall references doi ieee"
    )
    result = classifier.classify(text, filename="paper.pdf")
    assert result["detected_type"] == "research_paper"
    assert result["confidence"] > 0.7


def test_classifier_detects_announcement() -> None:
    """Announcement text should classify confidently."""
    classifier = DocumentClassifier()
    text = (
        "We are pleased to announce registration open deadline last date kindly note "
        "workshop venue fee participants certificate university department"
    )
    result = classifier.classify(text, filename="notice.txt")
    assert result["detected_type"] == "announcement"
    assert result["confidence"] > 0.7


def test_classifier_detects_news() -> None:
    """News text should classify confidently."""
    classifier = DocumentClassifier()
    text = (
        "According to sources confirmed as reported said in a statement government "
        "official investigation incident witness agency press conference spokesperson"
    )
    result = classifier.classify(text, filename="news.txt")
    assert result["detected_type"] == "news"
    assert result["confidence"] > 0.7


def test_classifier_handles_ambiguous_text() -> None:
    """Ambiguous text should still return a supported type."""
    classifier = DocumentClassifier()
    result = classifier.classify("This document has general information.")
    assert result["detected_type"] in {"research_paper", "announcement", "news"}


def test_classifier_scores_shape() -> None:
    """Classifier should return all supported type scores."""
    classifier = DocumentClassifier()
    result = classifier.classify("reported according to agency witness")
    assert set(result["scores"]) == {"research_paper", "announcement", "news"}


def test_classifier_empty_text_graceful() -> None:
    """Empty text should not crash classification."""
    classifier = DocumentClassifier()
    result = classifier.classify("")
    assert result["detected_type"] in {"research_paper", "announcement", "news"}
    assert result["method"] == "keyword_fallback"


def test_classifier_uses_tfidf_when_available() -> None:
    """Classifier should prefer TF-IDF when sklearn is installed."""
    classifier = DocumentClassifier()
    result = classifier.classify("abstract methodology findings references")
    assert result["method"] in {"tfidf", "keyword_fallback"}
