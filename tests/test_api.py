"""API endpoint tests."""

from __future__ import annotations

import io

import fitz

from app.models.summarizer import SummaryTimeoutError


def _build_pdf_bytes(text: str) -> bytes:
    """Create in-memory PDF bytes for testing.

    Args:
        text: Text to place in the PDF.

    Returns:
        PDF bytes.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_health_endpoint(client) -> None:
    """Health endpoint returns status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert "timestamp_utc" in payload


def test_summarize_txt_success(client, monkeypatch) -> None:
    """Summarize TXT upload and return summaries + evaluation."""

    def fake_generate_summaries(text: str, timeout_seconds: int):
        return {
            "summary_bart": "bart summary",
            "summary_pegasus": "pegasus summary",
            "summary_t5": "t5 summary",
        }

    def fake_evaluate(source_text: str, summaries):
        return {
            "pairwise_rouge": {
                "summary_bart_vs_summary_pegasus": {"rouge1": 0.2, "rouge2": 0.12, "rougeL": 0.21}
            },
            "custom_metrics": {},
        }

    monkeypatch.setattr("app.routes.api.summarizer.generate_summaries", fake_generate_summaries)
    monkeypatch.setattr("app.routes.api.evaluator.evaluate_summaries", fake_evaluate)

    data = {
        "file": (io.BytesIO(b"This is a short sample news article for summary testing."), "sample.txt"),
    }
    response = client.post("/api/summarize", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] in {"news", "announcement", "research_paper"}
    assert payload["summaries"]["summary_bart"] == "bart summary"
    assert "evaluation" in payload


def test_summarize_pdf_success(client, monkeypatch) -> None:
    """Summarize PDF upload with mocked model generation."""

    monkeypatch.setattr(
        "app.routes.api.summarizer.generate_summaries",
        lambda text, timeout_seconds: {
            "summary_bart": "bart pdf",
            "summary_pegasus": "pegasus pdf",
            "summary_t5": "t5 pdf",
        },
    )
    monkeypatch.setattr(
        "app.routes.api.evaluator.evaluate_summaries",
        lambda source_text, summaries: {"pairwise_rouge": {}, "custom_metrics": {}},
    )

    pdf_bytes = _build_pdf_bytes("Abstract This paper proposes a robust testing method.")
    response = client.post(
        "/api/summarize",
        data={"file": (io.BytesIO(pdf_bytes), "paper.pdf")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] == "research_paper"


def test_invalid_file_extension(client) -> None:
    """Reject unsupported file extensions."""
    response = client.post(
        "/api/summarize",
        data={"file": (io.BytesIO(b"hello"), "sample.docx")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "supported" in response.get_json()["error"].lower()


def test_model_timeout_error_handling(client, monkeypatch) -> None:
    """Return 504 when model generation exceeds timeout."""

    def _raise_timeout(text: str, timeout_seconds: int):
        raise SummaryTimeoutError("Model summarization timed out after 60.0s.")

    monkeypatch.setattr("app.routes.api.summarizer.generate_summaries", _raise_timeout)

    response = client.post(
        "/api/summarize",
        data={"file": (io.BytesIO(b"breaking reported witness agency"), "news.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 504
    assert "timed out" in response.get_json()["error"].lower()
