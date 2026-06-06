"""Citation extraction helpers for research documents."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Set

LOGGER = logging.getLogger(__name__)


class CitationExtractor:
    """Extract lightweight citation metadata from research papers."""

    _VENUES = (
        "IEEE",
        "ACM",
        "CVPR",
        "NeurIPS",
        "NIPS",
        "ICML",
        "ICLR",
        "ACL",
        "EMNLP",
        "AAAI",
        "IJCAI",
        "KDD",
        "SIGIR",
        "WWW",
        "arXiv",
        "Nature",
        "Science",
    )
    _DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s,;)\]]+", re.IGNORECASE)
    _YEAR_PATTERN = re.compile(r"\b(19[9]\d|20[0-2]\d|2030)\b")
    _REFERENCE_PATTERN = re.compile(
        r"(\[\d+\]|\(\w+[^)]*,\s*(?:19[9]\d|20[0-2]\d|2030)\))"
    )

    def __init__(self) -> None:
        """Initialize lazy spaCy holder."""
        self._lock = threading.Lock()
        self._nlp: Any = None

    def _get_nlp(self) -> Any:
        """Load spaCy model on first use.

        Returns:
            spaCy language pipeline.
        """
        if self._nlp is not None:
            return self._nlp
        with self._lock:
            if self._nlp is not None:
                return self._nlp
            import spacy

            self._nlp = spacy.load("en_core_web_sm")
            return self._nlp

    @staticmethod
    def _unique_sorted(values: Set[str]) -> List[str]:
        """Return stable sorted string values.

        Args:
            values: String set.

        Returns:
            Sorted list.
        """
        return sorted(value for value in values if value)

    def _extract_authors(self, text: str) -> List[str]:
        """Extract PERSON entities from the top of the document.

        Args:
            text: Source document.

        Returns:
            Author candidates.
        """
        try:
            doc = self._get_nlp()(text[:500])
            authors = {
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ == "PERSON" and 2 <= len(ent.text.strip()) <= 80
            }
            return self._unique_sorted(authors)[:10]
        except Exception as error:
            LOGGER.warning("Author extraction failed: %s", error)
            return []

    def _extract_venues(self, text: str) -> List[str]:
        """Extract known venues and ORG entities.

        Args:
            text: Source document.

        Returns:
            Venue candidates.
        """
        found = {
            venue
            for venue in self._VENUES
            if re.search(rf"\b{re.escape(venue)}\b", text, re.IGNORECASE)
        }
        try:
            doc = self._get_nlp()(text[:4000])
            found.update(
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ == "ORG" and len(ent.text.strip()) <= 80
            )
        except Exception as error:
            LOGGER.warning("Venue entity extraction failed: %s", error)
        return self._unique_sorted(found)[:20]

    def _citation_count(self, text: str) -> int:
        """Estimate number of references in a paper.

        Args:
            text: Source document.

        Returns:
            Non-negative citation count.
        """
        references_start = re.search(r"\breferences\b", text, re.IGNORECASE)
        reference_section = text[references_start.start() :] if references_start else text
        bracket_numbers = {
            int(match)
            for match in re.findall(r"\[(\d{1,3})\]", reference_section)
            if int(match) > 0
        }
        structured_matches = self._REFERENCE_PATTERN.findall(reference_section)
        return max(len(bracket_numbers), len(structured_matches))

    @staticmethod
    def _confidence(citation_count: int, dois: List[str]) -> str:
        """Convert citation evidence into a confidence label.

        Args:
            citation_count: Estimated citation count.
            dois: Extracted DOI list.

        Returns:
            high, medium, or low.
        """
        if dois or citation_count > 10:
            return "high"
        if citation_count >= 5:
            return "medium"
        return "low"

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract citation metadata from document text.

        Args:
            text: Source document text.

        Returns:
            Citation metadata payload.
        """
        source = text or ""
        dois = self._unique_sorted(set(self._DOI_PATTERN.findall(source)))
        years = sorted({int(year) for year in self._YEAR_PATTERN.findall(source)})
        citation_count = self._citation_count(source)

        payload = {
            "authors": self._extract_authors(source),
            "years_cited": years,
            "venues": self._extract_venues(source),
            "dois": dois,
            "citation_count": max(0, citation_count),
            "confidence": self._confidence(citation_count, dois),
        }
        return payload
