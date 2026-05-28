"""V2 API endpoints for caching and analytics."""

from __future__ import annotations

from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Optional

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import func

from app.db.schema import AnalyticsLog, create_session_factory, init_db
from app.models.evaluator import SummaryEvaluator
from app.models.summarizer import MultiModelSummarizer
from app.utils.cache import DocumentCache, compute_doc_hash
from app.utils.text_processor import (
    clean_text,
    detect_document_type,
    extract_text_from_upload,
    truncate_text,
)

api_v2_bp = Blueprint("api_v2", __name__, url_prefix="/api/v2")
summarizer = MultiModelSummarizer()
evaluator = SummaryEvaluator()


def _get_cache() -> DocumentCache:
    """Create cache instance from configured DB path.

    Returns:
        Document cache instance.
    """
    return DocumentCache(db_path=current_app.config["DB_PATH"])


def _get_analytics_session_factory():
    """Create analytics session factory for configured DB path.

    Returns:
        SQLAlchemy sessionmaker bound to configured DB.
    """
    db_path = current_app.config["DB_PATH"]
    init_db(db_path=db_path)
    return create_session_factory(db_path=db_path)


def _extract_primary_rouge1(evaluation: Dict[str, Any]) -> float:
    """Extract an aggregate ROUGE-1 value from evaluation payload.

    Args:
        evaluation: Evaluator output.

    Returns:
        Aggregate ROUGE-1 score.
    """
    pairwise = evaluation.get("pairwise_rouge", {})
    if not pairwise:
        return 0.0
    values = [pair.get("rouge1", 0.0) for pair in pairwise.values()]
    return float(sum(values) / len(values)) if values else 0.0


def _build_v2_response(
    summaries: Dict[str, str],
    rouge_scores: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Build standardized v2 summarize response payload.

    Args:
        summaries: Summaries from models.
        rouge_scores: Evaluation scores.
        metadata: Routing/cache/performance metadata.

    Returns:
        API response payload.
    """
    return {
        "summaries": summaries,
        "rouge_scores": rouge_scores,
        "metadata": metadata,
    }


@api_v2_bp.post("/summarize")
def summarize_v2() -> Any:
    """Summarize document with cache + analytics logging.

    Returns:
        JSON response with summaries, scores, and metadata.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Please attach a PDF or TXT file."}), HTTPStatus.BAD_REQUEST

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "No file selected. Please choose a PDF or TXT file."}), HTTPStatus.BAD_REQUEST

    requested_doc_type = request.form.get("document_type", "").strip().lower()
    raw_text = extract_text_from_upload(uploaded_file)
    normalized_text = clean_text(raw_text)
    truncated_text = truncate_text(
        normalized_text,
        max_words=current_app.config["MAX_INPUT_TOKENS"],
    )

    doc_hash = compute_doc_hash(truncated_text)
    cache = _get_cache()

    cached = cache.get(doc_hash)
    if cached is not None:
        cache.increment_hit_count(doc_hash)
        routed_summary_key, routed_model = summarizer.choose_routed_model(
            requested_doc_type or detect_document_type(truncated_text)
        )
        metadata = {
            "detected_doc_type": requested_doc_type or detect_document_type(truncated_text),
            "routed_model": routed_model,
            "was_cached": True,
            "inference_time_ms": 0.0,
            "doc_length_chars": len(truncated_text),
            "doc_hash": doc_hash,
            "recommended_summary_key": routed_summary_key,
        }
        return jsonify(
            _build_v2_response(
                summaries=cached["summaries"],
                rouge_scores=cached["rouge_scores"],
                metadata=metadata,
            )
        )

    detected_doc_type = requested_doc_type or detect_document_type(truncated_text)
    routed_summary_key, routed_model = summarizer.choose_routed_model(detected_doc_type)

    start = monotonic()
    summaries = summarizer.generate_summaries(
        text=truncated_text,
        timeout_seconds=current_app.config["MODEL_TIMEOUT_SECONDS"],
        use_parallel=summarizer.should_use_parallel(),
    )
    evaluation = evaluator.evaluate_summaries(source_text=truncated_text, summaries=summaries)
    inference_time_ms = round((monotonic() - start) * 1000, 2)

    cache.set(doc_hash=doc_hash, summaries=summaries, rouge_scores=evaluation)
    rouge_1 = _extract_primary_rouge1(evaluation=evaluation)

    session_factory = _get_analytics_session_factory()
    with session_factory() as session:
        session.add(
            AnalyticsLog(
                doc_type=detected_doc_type,
                model_selected=routed_model,
                rouge_1=rouge_1,
                inference_time_ms=float(inference_time_ms),
                was_cached=0,
                created_at=datetime.now(timezone.utc),
            )
        )
        session.commit()

    metadata = {
        "detected_doc_type": detected_doc_type,
        "routed_model": routed_model,
        "was_cached": False,
        "inference_time_ms": inference_time_ms,
        "doc_length_chars": len(truncated_text),
        "doc_hash": doc_hash,
        "recommended_summary_key": routed_summary_key,
    }

    return jsonify(_build_v2_response(summaries=summaries, rouge_scores=evaluation, metadata=metadata))


@api_v2_bp.get("/analytics")
def get_analytics() -> Any:
    """Return aggregated analytics overview.

    Returns:
        JSON payload with summary stats and grouped metrics.
    """
    session_factory = _get_analytics_session_factory()
    cache = _get_cache()
    cache_stats = cache.get_stats()

    with session_factory() as session:
        total_documents = session.query(func.count(AnalyticsLog.id)).scalar() or 0
        avg_latency = session.query(func.avg(AnalyticsLog.inference_time_ms)).scalar() or 0.0

        doc_type_rows = (
            session.query(
                AnalyticsLog.doc_type,
                func.count(AnalyticsLog.id),
                func.avg(AnalyticsLog.rouge_1),
            )
            .group_by(AnalyticsLog.doc_type)
            .all()
        )
        by_doc_type: Dict[str, Dict[str, float]] = {
            row[0]: {"count": float(row[1]), "avg_rouge_1": round(float(row[2] or 0.0), 4)}
            for row in doc_type_rows
        }

        model_rows = (
            session.query(
                AnalyticsLog.model_selected,
                func.count(AnalyticsLog.id),
                func.avg(AnalyticsLog.rouge_1),
                func.avg(AnalyticsLog.inference_time_ms),
            )
            .group_by(AnalyticsLog.model_selected)
            .all()
        )
        by_model: Dict[str, Dict[str, float]] = {
            row[0]: {
                "count": float(row[1]),
                "avg_rouge_1": round(float(row[2] or 0.0), 4),
                "avg_latency_ms": round(float(row[3] or 0.0), 2),
            }
            for row in model_rows
        }

        recent_rows = (
            session.query(AnalyticsLog)
            .order_by(AnalyticsLog.created_at.desc())
            .limit(20)
            .all()
        )
        recent_summaries: List[Dict[str, Any]] = [
            {
                "timestamp": row.created_at.isoformat(),
                "doc_type": row.doc_type,
                "model": row.model_selected,
                "rouge_1": round(float(row.rouge_1), 4),
                "latency_ms": round(float(row.inference_time_ms), 2),
                "was_cached": bool(row.was_cached),
            }
            for row in recent_rows
        ]

    payload = {
        "summary": {
            "total_documents": float(total_documents),
            "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
            "avg_inference_time_ms": round(float(avg_latency), 2),
        },
        "by_doc_type": by_doc_type,
        "by_model": by_model,
        "recent_summaries": recent_summaries,
    }
    return jsonify(payload)


@api_v2_bp.post("/cache/clear")
def clear_cache() -> Any:
    """Clear one or all cache entries.

    Returns:
        JSON payload with cleared count and freed bytes.
    """
    payload: Optional[Dict[str, Any]] = request.get_json(silent=True)
    doc_hash = payload.get("doc_hash") if payload else None

    cache = _get_cache()
    db_path = Path(current_app.config["DB_PATH"])
    before = db_path.stat().st_size if db_path.exists() else 0
    cleared = cache.clear(doc_hash=doc_hash)
    after = db_path.stat().st_size if db_path.exists() else 0
    freed_bytes = before - after if before >= after else 0

    return jsonify({"cleared": int(cleared), "freed_bytes": int(freed_bytes)})

