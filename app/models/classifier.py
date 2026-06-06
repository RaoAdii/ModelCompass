"""Document classification utilities for intelligent routing."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Tuple

LOGGER = logging.getLogger(__name__)

DOCUMENT_TYPES = ("research_paper", "announcement", "news")


class DocumentClassifier:
    """Classify documents with TF-IDF and keyword fallback."""

    _HIGH_WEIGHT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "research_paper": (
            "abstract",
            "methodology",
            "hypothesis",
            "empirical",
            "findings",
            "literature review",
            "proposed method",
            "experimental results",
        ),
        "announcement": (
            "we are pleased to announce",
            "effective immediately",
            "registration open",
            "deadline",
            "last date",
            "kindly note",
            "notice",
        ),
        "news": (
            "according to",
            "sources confirmed",
            "as reported",
            "said in a statement",
            "confirmed by",
        ),
    }
    _NORMAL_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "research_paper": (
            "introduction",
            "related work",
            "conclusion",
            "references",
            "dataset",
            "baseline",
            "evaluation",
            "accuracy",
            "precision",
            "recall",
            "table",
            "figure",
            "equation",
            "theorem",
            "proof",
            "doi",
            "arxiv",
            "ieee",
        ),
        "announcement": (
            "event",
            "workshop",
            "seminar",
            "fee",
            "venue",
            "schedule",
            "participants",
            "register",
            "certificate",
            "srm",
            "university",
            "college",
            "department",
            "principal",
            "director",
            "invited",
            "attendance",
        ),
        "news": (
            "reported",
            "breaking",
            "exclusive",
            "witness",
            "agency",
            "correspondent",
            "press conference",
            "spokesperson",
            "government",
            "official",
            "investigation",
            "incident",
        ),
    }

    def __init__(self) -> None:
        """Initialize classifier resources lazily."""
        self._tfidf_ready = False
        self._vectorizer: Any = None
        self._seed_matrix: Any = None
        self._seed_labels: List[str] = []
        self._load_tfidf()

    def _load_tfidf(self) -> None:
        """Prepare TF-IDF seed vectors when scikit-learn is installed."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            seed_docs: List[str] = []
            seed_labels: List[str] = []
            for doc_type in DOCUMENT_TYPES:
                phrases = list(self._HIGH_WEIGHT_KEYWORDS[doc_type]) * 2
                phrases.extend(self._NORMAL_KEYWORDS[doc_type])
                seed_docs.append(" ".join(phrases))
                seed_labels.append(doc_type)

            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                stop_words="english",
                lowercase=True,
            )
            self._seed_matrix = self._vectorizer.fit_transform(seed_docs)
            self._seed_labels = seed_labels
            self._tfidf_ready = True
        except Exception as error:
            LOGGER.warning("TF-IDF classifier unavailable, using keyword fallback: %s", error)
            self._tfidf_ready = False

    @staticmethod
    def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize raw scores to probabilities.

        Args:
            scores: Raw score by document type.

        Returns:
            Scores normalized to sum to one.
        """
        total = sum(max(value, 0.0) for value in scores.values())
        if total <= 0:
            return {doc_type: round(1.0 / len(DOCUMENT_TYPES), 4) for doc_type in DOCUMENT_TYPES}
        return {
            doc_type: round(max(scores.get(doc_type, 0.0), 0.0) / total, 4)
            for doc_type in DOCUMENT_TYPES
        }

    @staticmethod
    def _count_phrase(text: str, phrase: str) -> int:
        """Count phrase occurrences with word boundaries where possible.

        Args:
            text: Lowercase input text.
            phrase: Keyword or phrase to count.

        Returns:
            Number of occurrences.
        """
        escaped = re.escape(phrase.lower())
        if " " in phrase:
            return len(re.findall(escaped, text))
        return len(re.findall(rf"\b{escaped}\b", text))

    def _keyword_scores(self, text: str, filename: str = "") -> Dict[str, float]:
        """Compute weighted keyword scores.

        Args:
            text: Source document text.
            filename: Optional uploaded filename.

        Returns:
            Raw score by document type.
        """
        haystack = f"{filename} {text}".lower()
        scores: Dict[str, float] = {}
        for doc_type in DOCUMENT_TYPES:
            high = sum(
                2.0 * self._count_phrase(haystack, keyword)
                for keyword in self._HIGH_WEIGHT_KEYWORDS[doc_type]
            )
            normal = sum(
                1.0 * self._count_phrase(haystack, keyword)
                for keyword in self._NORMAL_KEYWORDS[doc_type]
            )
            filename_bonus = 1.5 if doc_type.replace("_", "") in filename.lower() else 0.0
            scores[doc_type] = high + normal + filename_bonus
        return scores

    def _tfidf_scores(self, text: str, filename: str = "") -> Dict[str, float]:
        """Compute TF-IDF cosine similarity scores.

        Args:
            text: Source document text.
            filename: Optional uploaded filename.

        Returns:
            Similarity score by document type.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        query = self._vectorizer.transform([f"{filename} {text}"])
        similarities = cosine_similarity(query, self._seed_matrix)[0]
        keyword_scores = self._normalize_scores(self._keyword_scores(text, filename))
        scores: Dict[str, float] = {}
        for label, similarity in zip(self._seed_labels, similarities):
            scores[label] = float(similarity) + (0.35 * keyword_scores[label])
        return scores

    @staticmethod
    def _pick_type(scores: Dict[str, float]) -> Tuple[str, float]:
        """Choose best document type and confidence.

        Args:
            scores: Normalized scores by document type.

        Returns:
            Detected type and confidence.
        """
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        detected_type = ranked[0][0]
        confidence = ranked[0][1]
        return detected_type, round(float(confidence), 4)

    def classify(self, text: str, filename: str = "") -> Dict[str, Any]:
        """Classify a document into a supported type.

        Args:
            text: Document text.
            filename: Optional uploaded filename.

        Returns:
            Classification payload with type, confidence, scores, and method.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            scores = {doc_type: round(1.0 / len(DOCUMENT_TYPES), 4) for doc_type in DOCUMENT_TYPES}
            return {
                "detected_type": "news",
                "confidence": scores["news"],
                "scores": scores,
                "method": "keyword_fallback",
            }

        if self._tfidf_ready:
            try:
                scores = self._normalize_scores(self._tfidf_scores(cleaned, filename))
                detected_type, confidence = self._pick_type(scores)
                return {
                    "detected_type": detected_type,
                    "confidence": confidence,
                    "scores": scores,
                    "method": "tfidf",
                }
            except Exception as error:
                LOGGER.warning("TF-IDF classification failed, falling back: %s", error)

        scores = self._normalize_scores(self._keyword_scores(cleaned, filename))
        detected_type, confidence = self._pick_type(scores)
        return {
            "detected_type": detected_type,
            "confidence": confidence,
            "scores": scores,
            "method": "keyword_fallback",
        }
