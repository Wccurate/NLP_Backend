"""Text processing helpers for documents and prompts."""

from __future__ import annotations

import re
from typing import List

DOCUMENT_PATTERN = re.compile(r"<document>(.*?)</document>", re.DOTALL | re.IGNORECASE)


def extract_document(input_str: str) -> str:
    """Return the first document block from the input, if present."""

    match = DOCUMENT_PATTERN.search(input_str)
    return match.group(1).strip() if match else ""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks suitable for retrieval."""

    if not text:
        return []
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunks.append(text[start:end])
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


def strip_document_tags(input_str: str) -> str:
    """Remove document tags from the string."""

    return DOCUMENT_PATTERN.sub("", input_str).strip()

