"""Custom exceptions for ModelCompass."""

from __future__ import annotations


class SummaryTimeoutError(TimeoutError):
    """Raised when summarization exceeds allowed timeout."""
