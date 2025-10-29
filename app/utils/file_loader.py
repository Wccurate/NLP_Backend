"""File loading helpers for uploaded documents."""

from __future__ import annotations

import io
import logging
from pathlib import Path

from fastapi import UploadFile
from pypdf import PdfReader
from docx import Document

logger = logging.getLogger(__name__)


async def extract_text_from_upload(upload: UploadFile | None) -> str:
    """Extract text content from an uploaded file."""

    if upload is None:
        return ""

    try:
        data = await upload.read()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed reading uploaded file: %s", exc)
        return ""
    finally:
        await upload.close()

    if not data:
        return ""

    suffix = Path(upload.filename or "").suffix.lower()
    content_type = (upload.content_type or "").lower()

    if suffix == ".docx" or "word" in content_type:
        return _extract_docx(data)
    if suffix == ".pdf" or "pdf" in content_type:
        return _extract_pdf(data)
    if content_type.startswith("text/"):
        return data.decode("utf-8", errors="ignore")

    try:
        return data.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        logger.warning("Could not decode upload %s; returning empty string", upload.filename)
        return ""


def _extract_docx(data: bytes) -> str:
    try:
        document = Document(io.BytesIO(data))
        paragraphs = [para.text.strip() for para in document.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DOCX extraction failed: %s", exc)
        return ""


def _extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        text_parts = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            extracted = extracted.strip()
            if extracted:
                text_parts.append(extracted)
        return "\n".join(text_parts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF extraction failed: %s", exc)
        return ""

