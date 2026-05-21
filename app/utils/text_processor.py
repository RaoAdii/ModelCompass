"""Text processing helpers for uploaded documents."""

from __future__ import annotations

import io
import re
from typing import BinaryIO

import fitz
from werkzeug.datastructures import FileStorage

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def _get_extension(filename: str) -> str:
    """Extract lowercase file extension.

    Args:
        filename: Uploaded filename.

    Returns:
        Lowercased extension (including dot), or empty string.
    """
    if "." not in filename:
        return ""
    return f".{filename.rsplit('.', 1)[1].lower()}"


def validate_file_extension(filename: str) -> None:
    """Validate allowed upload extension.

    Args:
        filename: Uploaded filename.

    Raises:
        ValueError: If extension is not allowed.
    """
    extension = _get_extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only PDF and TXT files are supported.")


def extract_text_from_pdf(binary_stream: BinaryIO) -> str:
    """Extract text from a PDF binary stream.

    Args:
        binary_stream: Stream containing PDF bytes.

    Returns:
        Extracted raw text.

    Raises:
        ValueError: If PDF cannot be parsed.
    """
    try:
        content = binary_stream.read()
        if not content:
            raise ValueError("Uploaded PDF is empty.")
        with fitz.open(stream=content, filetype="pdf") as doc:
            pages = [page.get_text("text") for page in doc]
    except Exception as error:
        raise ValueError("Invalid PDF file. Please upload a valid PDF.") from error

    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("No readable text found in PDF.")
    return text


def extract_text_from_txt(binary_stream: BinaryIO) -> str:
    """Extract text from a UTF-8 text stream.

    Args:
        binary_stream: Stream containing text bytes.

    Returns:
        Decoded text.

    Raises:
        ValueError: If file is empty.
    """
    text = binary_stream.read().decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("Uploaded text file is empty.")
    return text


def clean_text(text: str) -> str:
    """Normalize document text.

    Args:
        text: Raw text.

    Returns:
        Cleaned text with normalized whitespace.
    """
    normalized = text.replace("\x00", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def truncate_text(text: str, max_words: int) -> str:
    """Truncate text to a max word limit.

    Args:
        text: Input text.
        max_words: Maximum word count to keep.

    Returns:
        Truncated text.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def detect_document_type(text: str) -> str:
    """Heuristically detect document type.

    Args:
        text: Cleaned document text.

    Returns:
        One of research_paper, announcement, news.
    """
    lowered = text.lower()
    research_hits = sum(
        keyword in lowered
        for keyword in ("abstract", "methodology", "results", "conclusion", "references")
    )
    announcement_hits = sum(
        keyword in lowered
        for keyword in ("announce", "launch", "effective immediately", "we are pleased", "official")
    )
    news_hits = sum(
        keyword in lowered
        for keyword in ("reported", "according to", "breaking", "witness", "agency")
    )

    if research_hits >= max(announcement_hits, news_hits):
        return "research_paper"
    if announcement_hits >= max(research_hits, news_hits):
        return "announcement"
    return "news"


def extract_text_from_upload(uploaded_file: FileStorage) -> str:
    """Extract text from an uploaded file based on extension.

    Args:
        uploaded_file: Flask uploaded file object.

    Returns:
        Extracted raw text.
    """
    validate_file_extension(uploaded_file.filename or "")
    extension = _get_extension(uploaded_file.filename or "")
    uploaded_file.stream.seek(0)
    stream = io.BytesIO(uploaded_file.read())
    stream.seek(0)
    if extension == ".pdf":
        return extract_text_from_pdf(stream)
    return extract_text_from_txt(stream)
