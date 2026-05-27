"""Application factory for ModelCompass."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any, Dict

from flask import Flask, jsonify
from flask.wrappers import Response
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from app.config import Config
from app.exceptions import SummaryTimeoutError
from app.routes.api import api_bp


def _configure_logging(log_level: str) -> None:
    """Configure application logging.

    Args:
        log_level: Logging level name (e.g. INFO, DEBUG).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def create_app() -> Flask:
    """Create and configure the Flask app.

    Returns:
        Configured Flask app instance.
    """
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)
    _configure_logging(app.config["LOG_LEVEL"])
    app.register_blueprint(api_bp)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_file(_: RequestEntityTooLarge) -> Response:
        """Return a user-friendly response for oversized uploads."""
        response: Dict[str, Any] = {
            "error": "File too large. Please upload a smaller file.",
            "max_file_size_mb": app.config["MAX_FILE_SIZE_MB"],
        }
        return jsonify(response), HTTPStatus.REQUEST_ENTITY_TOO_LARGE

    @app.errorhandler(SummaryTimeoutError)
    def handle_timeout(error: SummaryTimeoutError) -> Response:
        """Return timeout response for summarization time budget failures."""
        response: Dict[str, str] = {"error": str(error)}
        return jsonify(response), HTTPStatus.GATEWAY_TIMEOUT

    @app.errorhandler(ValueError)
    def handle_value_error(error: ValueError) -> Response:
        """Return bad-request response for validation errors."""
        response: Dict[str, str] = {"error": str(error)}
        return jsonify(response), HTTPStatus.BAD_REQUEST

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception) -> Response:
        """Return generic internal server error payload."""
        app.logger.exception("Unhandled exception: %s", error)
        response: Dict[str, str] = {
            "error": "Unexpected server error. Please try again later."
        }
        return jsonify(response), HTTPStatus.INTERNAL_SERVER_ERROR

    return app
