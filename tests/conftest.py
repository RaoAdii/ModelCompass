"""Pytest fixtures for ModelCompass."""

from __future__ import annotations

import pytest

from app import create_app


@pytest.fixture()
def flask_app():
    """Create test Flask app."""
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
            "MAX_CONTENT_LENGTH": 2 * 1024 * 1024,
            "MODEL_TIMEOUT_SECONDS": 2,
            "MAX_INPUT_TOKENS": 256,
        }
    )
    return app


@pytest.fixture()
def client(flask_app):
    """Flask test client fixture."""
    return flask_app.test_client()
