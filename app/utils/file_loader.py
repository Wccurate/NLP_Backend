"""File loading helpers for uploaded documents."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Literal

import httpx
from docx import Document
from fastapi import UploadFile
from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

logger = logging.getLogger(__name__)


PdfStrategy = Literal["pymupdf", "pypdf", "ocr", "auto"]


async def extract_text_from_upload(
    upload: UploadFile | None, *, pdf_strategy: PdfStrategy = "pymupdf"
) -> str:
    """Extract text content from an uploaded file.

    Parameters
    ----------
    upload:
        The uploaded file from FastAPI.
    pdf_strategy:
        Which method to use for PDF extraction. Options:
        - "pymupdf": high-quality layout-aware extraction (default).
        - "pypdf": lightweight pure-python parser.
        - "ocr": send to external OCR API (requires OCR_SPACE_API_KEY).
        - "auto": try PyMuPDF, fall back to PyPDF, then OCR.
    """

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
        return await _extract_pdf(data, strategy=pdf_strategy)
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


async def _extract_pdf(data: bytes, *, strategy: PdfStrategy) -> str:
    strategies = (
        ["pymupdf", "pypdf", "ocr"]
        if strategy == "auto"
        else [strategy, "pymupdf", "pypdf", "ocr"]
    )

    seen = set()
    for method in strategies:
        if method in seen:
            continue
        seen.add(method)
        if method == "pymupdf":
            text = _extract_pdf_pymupdf(data)
            if text:
                return text
        elif method == "pypdf":
            text = _extract_pdf_pypdf(data)
            if text:
                return text
        elif method == "ocr":
            text = await _extract_pdf_ocr(data)
            if text:
                return text

    logger.warning("PDF extraction yielded no text with strategies %s", strategies)
    return ""


def _extract_pdf_pymupdf(data: bytes) -> str:
    if fitz is None:
        return ""
    try:
        document = fitz.open(stream=data, filetype="pdf")
        text_parts = []
        for page in document:
            extracted = page.get_text("text") or ""
            extracted = extracted.strip()
            if extracted:
                text_parts.append(extracted)
        return "\n".join(text_parts).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("PyMuPDF extraction failed: %s", exc)
        return ""


def _extract_pdf_pypdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        text_parts = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            extracted = extracted.strip()
            if extracted:
                text_parts.append(extracted)
        return "\n".join(text_parts).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("PyPDF extraction failed: %s", exc)
        return ""


async def _extract_pdf_ocr(data: bytes) -> str:
    api_key = os.getenv("OCR_SPACE_API_KEY")
    if not api_key:
        logger.warning("OCR_SPACE_API_KEY not set; skipping OCR extraction.")
        return ""

    payload = {"apikey": api_key, "language": os.getenv("OCR_SPACE_LANGUAGE", "eng")}
    files = {"file": ("document.pdf", data, "application/pdf")}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                os.getenv("OCR_SPACE_ENDPOINT", "https://api.ocr.space/parse/image"),
                data=payload,
                files=files,
            )
            resp.raise_for_status()
            result = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR API request failed: %s", exc)
        return ""

    if result.get("IsErroredOnProcessing"):
        logger.warning("OCR API reported error: %s", result.get("ErrorMessage"))
        return ""

    parsed_results = result.get("ParsedResults") or []
    text_parts = []
    for item in parsed_results:
        text = (item.get("ParsedText") or "").strip()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts).strip()
