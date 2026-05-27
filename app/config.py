"""Configuration for ModelCompass Flask app."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Runtime configuration loaded from environment variables."""

    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "5000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    MAX_CONTENT_LENGTH: int = MAX_FILE_SIZE_MB * 1024 * 1024
    MODEL_TIMEOUT_SECONDS: int = int(os.getenv("MODEL_TIMEOUT_SECONDS", "60"))
    MAX_INPUT_TOKENS: int = int(
        os.getenv("MAX_INPUT_TOKENS", os.getenv("MAX_INPUT_WORDS", "1024"))
    )

    BART_MODEL_NAME: str = os.getenv("BART_MODEL_NAME", "facebook/bart-large-cnn")
    PEGASUS_MODEL_NAME: str = os.getenv("PEGASUS_MODEL_NAME", "google/pegasus-xsum")
    T5_MODEL_NAME: str = os.getenv("T5_MODEL_NAME", "t5-small")

    SUMMARY_MAX_LENGTH: int = int(os.getenv("SUMMARY_MAX_LENGTH", "180"))
    SUMMARY_MIN_LENGTH: int = int(os.getenv("SUMMARY_MIN_LENGTH", "60"))
    USE_PARALLEL_ON_CPU: bool = os.getenv("USE_PARALLEL_ON_CPU", "false").lower() == "true"
    USE_PARALLEL_ON_GPU: bool = os.getenv("USE_PARALLEL_ON_GPU", "true").lower() == "true"
