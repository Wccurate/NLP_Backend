"""Pydantic schemas for request and response bodies."""

from __future__ import annotations

from typing import Dict, List, Optional

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


class JobDescriptionRequest(BaseModel):
    """Payload for adding job descriptions to the vector store."""

    text: str = Field(..., description="Plain text job description to index.")
    title: Optional[str] = Field(default=None, description="Optional title or identifier.")
    metadata: Optional[Dict[str, str]] = Field(
        default=None, description="Optional metadata key/value pairs."
    )


class JobDescriptionResponse(BaseModel):
    """Response after inserting job description chunks."""

    inserted: int
    ids: List[str] = Field(default_factory=list)
