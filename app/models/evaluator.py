"""Summary evaluation utilities based on ROUGE."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict

from rouge_score import rouge_scorer


class SummaryEvaluator:
    """Compute ROUGE and custom metrics for generated summaries."""

    def __init__(self) -> None:
        """Initialize ROUGE scorer."""
        self._scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )

    def _pairwise_rouge(self, summaries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Compute pairwise ROUGE F1 between all model summaries.

        Args:
            summaries: Summaries keyed by model name.

        Returns:
            Pairwise ROUGE map.
        """
        pairwise_scores: Dict[str, Dict[str, float]] = {}
        for left_key, right_key in combinations(sorted(summaries.keys()), 2):
            pair_name = f"{left_key}_vs_{right_key}"
            score = self._scorer.score(summaries[left_key], summaries[right_key])
            pairwise_scores[pair_name] = {
                "rouge1": round(score["rouge1"].fmeasure, 4),
                "rouge2": round(score["rouge2"].fmeasure, 4),
                "rougeL": round(score["rougeL"].fmeasure, 4),
            }
        return pairwise_scores

    @staticmethod
    def _custom_metrics(source_text: str, summaries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Compute custom metrics per summary.

        Args:
            source_text: Original document text.
            summaries: Summaries keyed by model name.

        Returns:
            Metrics including word count and compression ratio.
        """
        source_word_count = max(len(source_text.split()), 1)
        metrics: Dict[str, Dict[str, float]] = {}
        for key, summary in summaries.items():
            summary_word_count = len(summary.split())
            compression_ratio = round(summary_word_count / source_word_count, 4)
            metrics[key] = {
                "word_count": float(summary_word_count),
                "compression_ratio": compression_ratio,
            }
        return metrics

    def evaluate_summaries(self, source_text: str, summaries: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate generated summaries.

        Args:
            source_text: Original document text.
            summaries: Summaries keyed by model name.

        Returns:
            Evaluation payload with pairwise ROUGE and custom metrics.
        """
        return {
            "pairwise_rouge": self._pairwise_rouge(summaries),
            "custom_metrics": self._custom_metrics(source_text, summaries),
        }
