"""API routes for summarization and health checks."""

from __future__ import annotations

from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any, Dict

from flask import Blueprint, current_app, jsonify, request

from app.models.evaluator import SummaryEvaluator
from app.models.summarizer import MultiModelSummarizer
from app.utils.text_processor import clean_text, detect_document_type, extract_text_from_upload, truncate_text

api_bp = Blueprint("api", __name__, url_prefix="/api")
summarizer = MultiModelSummarizer()
evaluator = SummaryEvaluator()
VALID_DOCUMENT_TYPES = {"research_paper", "announcement", "news"}


@api_bp.get("/health")
def health() -> Any:
    """Health-check endpoint.

    Returns:
        API status information.
    """
    return jsonify(
        {
            "status": "ok",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )


@api_bp.post("/summarize")
def summarize_document() -> Any:
    """Summarize an uploaded document with three models.

    Returns:
        JSON response containing summaries and evaluation results.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Please attach a PDF or TXT file."}), HTTPStatus.BAD_REQUEST

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "No file selected. Please choose a PDF or TXT file."}), HTTPStatus.BAD_REQUEST

    requested_doc_type = request.form.get("document_type", "").strip().lower()
    if requested_doc_type and requested_doc_type not in VALID_DOCUMENT_TYPES:
        return (
            jsonify(
                {
                    "error": (
                        "Invalid document_type. Use one of: "
                        "research_paper, announcement, news."
                    )
                }
            ),
            HTTPStatus.BAD_REQUEST,
        )

    raw_text = extract_text_from_upload(uploaded_file)
    normalized_text = clean_text(raw_text)
    truncated_text = truncate_text(
        normalized_text,
        max_words=current_app.config["MAX_INPUT_WORDS"],
    )

    detected_doc_type = requested_doc_type or detect_document_type(truncated_text)
    routed_summary_key, routed_model = summarizer.choose_routed_model(detected_doc_type)

    summaries = summarizer.generate_summaries(
        text=truncated_text,
        timeout_seconds=current_app.config["MODEL_TIMEOUT_SECONDS"],
    )
    evaluation = evaluator.evaluate_summaries(
        source_text=truncated_text,
        summaries=summaries,
    )

    payload: Dict[str, Any] = {
        "document_type": detected_doc_type,
        "routed_model": routed_model,
        "summaries": summaries,
        "recommended_summary": summaries[routed_summary_key],
        "evaluation": evaluation,
    }
    return jsonify(payload), HTTPStatus.OK
