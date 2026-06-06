"""Tests for research citation extraction."""

from __future__ import annotations

from app.utils.citation_extractor import CitationExtractor


class _FakeEntity:
    """Fake spaCy entity."""

    def __init__(self, text: str, label: str) -> None:
        """Store fake entity fields."""
        self.text = text
        self.label_ = label


class _FakeDoc:
    """Fake spaCy doc."""

    def __init__(self, ents) -> None:
        """Store fake entities."""
        self.ents = ents


def test_extract_doi_from_paper_text(monkeypatch) -> None:
    """Extractor should find DOI patterns."""
    extractor = CitationExtractor()
    monkeypatch.setattr(extractor, "_get_nlp", lambda: lambda text: _FakeDoc([]))
    result = extractor.extract("This paper has DOI 10.1145/3394486.3403084 and references.")
    assert "10.1145/3394486.3403084" in result["dois"]
    assert result["confidence"] == "high"


def test_extract_years_from_citation_list(monkeypatch) -> None:
    """Extractor should capture citation years."""
    extractor = CitationExtractor()
    monkeypatch.setattr(extractor, "_get_nlp", lambda: lambda text: _FakeDoc([]))
    result = extractor.extract("References [1] Smith 2019. [2] Doe 2021. Brown 2022.")
    assert {2019, 2021, 2022}.issubset(set(result["years_cited"]))


def test_low_confidence_for_few_citations(monkeypatch) -> None:
    """Fewer than five citations without DOI should be low confidence."""
    extractor = CitationExtractor()
    monkeypatch.setattr(extractor, "_get_nlp", lambda: lambda text: _FakeDoc([]))
    result = extractor.extract("References [1] Smith 2019. [2] Doe 2021.")
    assert result["confidence"] == "low"


def test_non_research_text_empty_gracefully(monkeypatch) -> None:
    """Plain text should produce empty metadata without crashing."""
    extractor = CitationExtractor()
    monkeypatch.setattr(extractor, "_get_nlp", lambda: lambda text: _FakeDoc([]))
    result = extractor.extract("A short event announcement with no references.")
    assert result["dois"] == []
    assert result["citation_count"] >= 0


def test_extract_authors_and_venues_with_mock(monkeypatch) -> None:
    """Extractor should use NER candidates when available."""
    extractor = CitationExtractor()

    def fake_nlp(text: str):
        return _FakeDoc([
            _FakeEntity("Jane Doe", "PERSON"),
            _FakeEntity("ACM", "ORG"),
        ])

    monkeypatch.setattr(extractor, "_get_nlp", lambda: fake_nlp)
    result = extractor.extract("Jane Doe. Proceedings of ACM. References [1] A 2020.")
    assert "Jane Doe" in result["authors"]
    assert "ACM" in result["venues"]


def test_citation_count_not_negative(monkeypatch) -> None:
    """Citation count should never be negative."""
    extractor = CitationExtractor()
    monkeypatch.setattr(extractor, "_get_nlp", lambda: lambda text: _FakeDoc([]))
    result = extractor.extract("")
    assert result["citation_count"] >= 0
