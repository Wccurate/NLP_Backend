"""Pydantic schemas for request and response bodies."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for text generation."""

    input: str = Field(..., description="User input text, may include <document> tags.")
    web_search: bool = False
    return_stream: bool = False
    persist_documents: bool = False


class SourceItem(BaseModel):
    """Metadata for a retrieved source."""

    source: str
    score: Optional[float] = None
    text: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response payload for generation results."""

    intent: str
    text: str
    sources: List[SourceItem] = Field(default_factory=list)
    tool_calls: List[str] = Field(default_factory=list)
