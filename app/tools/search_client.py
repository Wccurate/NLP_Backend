"""Search client abstractions for agent-controlled web retrieval."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Protocol

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)


class SearchProvider(Protocol):
    """Protocol for search providers."""

    def search(self, query: str, *, max_results: int = 5) -> List[Dict[str, str]]:
        """Return a list of search hits."""


class TavilySearch:
    """Search provider powered by Tavily's API."""

    def __init__(self, api_key: str, endpoint: str) -> None:
        self._endpoint = endpoint
        self._api_key = api_key

    def search(self, query: str, *, max_results: int = 5) -> List[Dict[str, str]]:
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
        }
        try:
            response = httpx.post(self._endpoint, json=payload, timeout=12.0)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily search failed: %s", exc)
            return []

        hits = data.get("results") or []
        results: List[Dict[str, str]] = []
        for item in hits[:max_results]:
            results.append(
                {
                    "title": item.get("title") or item.get("url") or "Result",
                    "url": item.get("url") or "",
                    "snippet": item.get("content") or item.get("snippet") or "",
                }
            )
        return results


class NullSearch:
    """Fallback provider when no search is configured."""

    def search(self, query: str, *, max_results: int = 5) -> List[Dict[str, str]]:
        logger.info("Search skipped (no provider configured) for query: %s", query)
        return []


_cached_client: SearchProvider | None = None


def get_search_client() -> SearchProvider:
    """Return a configured search provider."""

    global _cached_client
    if _cached_client is not None:
        return _cached_client

    settings = get_settings()
    provider = settings.primary_search_provider.lower()

    if provider == "tavily":
        api_key = settings.tavily_api_key
        if not api_key:
            logger.warning("TAVILY_API_KEY is not set; search will be disabled.")
            _cached_client = NullSearch()
        else:
            _cached_client = TavilySearch(api_key=api_key, endpoint=settings.tavily_endpoint)
    else:
        logger.warning("Unknown search provider '%s'; disabling search.", provider)
        _cached_client = NullSearch()

    return _cached_client

