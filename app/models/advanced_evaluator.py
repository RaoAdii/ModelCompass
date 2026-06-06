"""Advanced summary quality metrics."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


class AdvancedEvaluator:
    """Compute semantic, abstractive, and entity preservation metrics."""

    def __init__(self) -> None:
        """Initialize lazy model holders."""
        self._embedding_lock = threading.Lock()
        self._spacy_lock = threading.Lock()
        self._embedding_model: Any = None
        self._nlp: Any = None

    def _get_embedding_model(self) -> Any:
        """Load sentence-transformer model on first use.

        Returns:
            SentenceTransformer instance.
        """
        if self._embedding_model is not None:
            return self._embedding_model
        with self._embedding_lock:
            if self._embedding_model is not None:
                return self._embedding_model
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            return self._embedding_model

    def _get_nlp(self) -> Any:
        """Load spaCy model on first use.

        Returns:
            spaCy language pipeline.
        """
        if self._nlp is not None:
            return self._nlp
        with self._spacy_lock:
            if self._nlp is not None:
                return self._nlp
            import spacy

            self._nlp = spacy.load("en_core_web_sm")
            return self._nlp

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        """Compute cosine similarity for two vectors.

        Args:
            left: First embedding.
            right: Second embedding.

        Returns:
            Similarity clipped to 0..1.
        """
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0:
            return 0.0
        value = float(np.dot(left, right) / denominator)
        return round(max(0.0, min(1.0, value)), 4)

    def semantic_similarity(self, source_text: str, summary: str) -> float:
        """Measure semantic similarity between source and summary.

        Args:
            source_text: Original document.
            summary: Generated summary.

        Returns:
            Similarity score from 0 to 1.
        """
        if not source_text.strip() or not summary.strip():
            return 0.0
        model = self._get_embedding_model()
        embeddings = model.encode([source_text, summary], convert_to_numpy=True)
        return self._cosine_similarity(embeddings[0], embeddings[1])

    @staticmethod
    def _tokens(text: str) -> Tuple[str, ...]:
        """Tokenize text for n-gram metrics.

        Args:
            text: Input text.

        Returns:
            Lowercase alphanumeric tokens.
        """
        return tuple(re.findall(r"[a-z0-9]+", text.lower()))

    @classmethod
    def _bigrams(cls, text: str) -> Set[Tuple[str, str]]:
        """Build unique token bigrams.

        Args:
            text: Input text.

        Returns:
            Set of bigram tuples.
        """
        tokens = cls._tokens(text)
        return set(zip(tokens, tokens[1:]))

    def abstractiveness_score(self, source_text: str, summary: str) -> float:
        """Measure novel bigram ratio in the summary.

        Args:
            source_text: Original document.
            summary: Generated summary.

        Returns:
            Novel bigram ratio from 0 to 1.
        """
        summary_bigrams = self._bigrams(summary)
        if not summary_bigrams:
            return 0.0
        source_bigrams = self._bigrams(source_text)
        novel = summary_bigrams - source_bigrams
        return round(len(novel) / len(summary_bigrams), 4)

    @staticmethod
    def _entity_texts(doc: Any) -> Set[str]:
        """Extract normalized entities of interest from a spaCy doc.

        Args:
            doc: spaCy Doc.

        Returns:
            Entity text set.
        """
        labels = {"PERSON", "ORG", "DATE", "LOC", "GPE"}
        return {
            entity.text.strip().lower()
            for entity in getattr(doc, "ents", [])
            if entity.label_ in labels and entity.text.strip()
        }

    def entity_preservation_score(self, source_text: str, summary: str) -> float:
        """Measure how many source entities are preserved in the summary.

        Args:
            source_text: Original document.
            summary: Generated summary.

        Returns:
            Entity preservation score from 0 to 1.
        """
        nlp = self._get_nlp()
        source_entities = self._entity_texts(nlp(source_text[:5000]))
        if not source_entities:
            return 1.0
        summary_entities = self._entity_texts(nlp(summary[:3000]))
        preserved = source_entities.intersection(summary_entities)
        return round(len(preserved) / len(source_entities), 4)

    @staticmethod
    def _safe_metric(metric_name: str, callback: Any) -> Optional[float]:
        """Run a metric and return None on failure.

        Args:
            metric_name: Name used in warning logs.
            callback: Zero-argument callable.

        Returns:
            Metric value or None.
        """
        try:
            return float(callback())
        except Exception as error:
            LOGGER.warning("Advanced metric %s failed: %s", metric_name, error)
            return None

    def evaluate_all(self, source_text: str, summaries: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate all advanced metrics for each summary.

        Args:
            source_text: Original document.
            summaries: Summaries keyed by model summary key.

        Returns:
            Advanced metrics keyed by summary key.
        """
        results: Dict[str, Any] = {}
        for key, summary in summaries.items():
            results[key] = {
                "semantic_similarity": self._safe_metric(
                    "semantic_similarity",
                    lambda source_text=source_text, summary=summary: self.semantic_similarity(
                        source_text,
                        summary,
                    ),
                ),
                "abstractiveness": self._safe_metric(
                    "abstractiveness",
                    lambda source_text=source_text, summary=summary: self.abstractiveness_score(
                        source_text,
                        summary,
                    ),
                ),
                "entity_preservation": self._safe_metric(
                    "entity_preservation",
                    lambda source_text=source_text, summary=summary: self.entity_preservation_score(
                        source_text,
                        summary,
                    ),
                ),
            }
        return results
