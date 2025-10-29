"""Simple DuckDuckGo search helper."""

from __future__ import annotations

import logging
from typing import Dict, List

import httpx

logger = logging.getLogger(__name__)

DUCKDUCKGO_URL = "https://duckduckgo.com/"


async def web_search(query: str, *, timeout: float = 6.0) -> List[Dict[str, str]]:
    """Perform a search request and return lightweight snippets."""

    params = {"q": query, "format": "json", "no_redirect": "1", "skip_disambig": "1"}
    results: List[Dict[str, str]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(DUCKDUCKGO_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("web_search error: %s", exc)
            return results

    # Extract from RelatedTopics or Abstract
    abstract = payload.get("AbstractText")
    if abstract:
        results.append(
            {
                "title": payload.get("Heading") or "DuckDuckGo",
                "url": payload.get("AbstractURL") or "",
                "snippet": abstract,
            }
        )

    related = payload.get("RelatedTopics") or []
    for item in related:
        if isinstance(item, dict) and item.get("Text"):
            results.append(
                {
                    "title": item.get("Text", "")[:120],
                    "url": item.get("FirstURL", ""),
                    "snippet": item.get("Text", ""),
                }
            )
        if len(results) >= 5:
            break

    return results

