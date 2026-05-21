"""Evaluator tests."""

from __future__ import annotations

from app.models.evaluator import SummaryEvaluator


def test_pairwise_rouge_reasonable_scores() -> None:
    """ROUGE scores should be non-trivial for related summaries."""
    evaluator = SummaryEvaluator()
    source = (
        "The committee announced a clean energy investment package. "
        "The proposal includes grants, tax incentives, and a three-year rollout."
    )
    summaries = {
        "summary_bart": "The committee announced a clean energy package with grants and tax incentives.",
        "summary_pegasus": "A clean energy investment plan was announced with incentives and grants over three years.",
        "summary_t5": "Officials announced an energy package including grants and tax incentives.",
    }
    evaluation = evaluator.evaluate_summaries(source_text=source, summaries=summaries)

    for pair_score in evaluation["pairwise_rouge"].values():
        assert pair_score["rouge1"] > 0.1
        assert pair_score["rougeL"] > 0.1
