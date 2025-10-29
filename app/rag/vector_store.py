"""Utilities for interacting with the Chroma vector store and BM25 fallback."""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from ..config import get_settings
from ..llm_provider import ChatProvider, get_llm_provider

logger = logging.getLogger(__name__)

COLLECTION_NAME = "jobs_demo"

_DEFAULT_CORPUS: List[Dict[str, str]] = [
    {
        "id": "jobs_demo#1",
        "text": "Software Engineer working on backend APIs with FastAPI and cloud services.",
    },
    {
        "id": "jobs_demo#2",
        "text": "Machine Learning Engineer focusing on NLP, transformers, and production pipelines.",
    },
    {
        "id": "jobs_demo#3",
        "text": "Data Analyst role requiring SQL, dashboards, and storytelling with data.",
    },
    {
        "id": "jobs_demo#4",
        "text": "DevOps Engineer supporting CI/CD, Kubernetes, and infrastructure automation.",
    },
]


@lru_cache()
def _get_client() -> chromadb.api.client.ClientAPI:
    settings = get_settings()
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    client_settings = Settings(anonymized_telemetry=False, allow_reset=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir), settings=client_settings)


@lru_cache()
def get_collection() -> Collection:
    """Return the persistent Chroma collection."""

    client = _get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _ensure_ids(count: int) -> List[str]:
    return [str(uuid.uuid4()) for _ in range(count)]


def add_texts(
    texts: Sequence[str],
    metadatas: Optional[Sequence[Dict[str, str]]] = None,
    *,
    ids: Optional[Sequence[str]] = None,
    embedder: Optional[ChatProvider] = None,
) -> List[str]:
    """Add documents to Chroma and return their ids."""

    if not texts:
        return []
    embedder = embedder or get_llm_provider()
    col = get_collection()
    doc_ids = list(ids) if ids else _ensure_ids(len(texts))
    embeddings = embedder.embed(texts)
    col.add(
        ids=doc_ids,
        documents=list(texts),
        metadatas=list(metadatas) if metadatas else None,
        embeddings=embeddings,
    )
    return doc_ids


def delete(ids: Iterable[str]) -> None:
    """Delete documents by id."""

    col = get_collection()
    try:
        col.delete(ids=list(ids))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed deleting ids %s: %s", ids, exc)


def _bm25_search(
    query: str,
    *,
    docs: Optional[Sequence[Dict[str, str]]] = None,
    top_k: int = 4,
) -> List[Dict[str, str]]:
    corpus = list(docs) if docs else _DEFAULT_CORPUS
    if not corpus:
        return []
    tokenized = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    scored = sorted(
        zip(corpus, scores),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]
    return [
        {
            "id": entry["id"],
            "text": entry["text"],
            "score": float(score),
            "metadata": entry.get("metadata", {}),
            "source": entry["id"],
        }
        for entry, score in scored
    ]


def search(
    query_text: str,
    *,
    top_k: int = 4,
    hyde_text: Optional[str] = None,
    extra_corpus: Optional[Sequence[Dict[str, str]]] = None,
    embedder: Optional[ChatProvider] = None,
) -> List[Dict[str, object]]:
    """Hybrid search combining Chroma and BM25 results."""

    embedder = embedder or get_llm_provider()
    col = get_collection()
    dense_results: List[Dict[str, object]] = []
    try:
        query_embedding = embedder.embed([hyde_text or query_text])[0]
        res = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        for doc_id, doc, meta, distance in zip(ids, docs, metas, distances):
            dense_results.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta or {},
                    "score": float(1 - distance) if distance is not None else None,
                    "source": meta.get("source") if isinstance(meta, dict) else doc_id,
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector search failed: %s", exc)

    bm25_docs: List[Dict[str, str]] = list(extra_corpus) if extra_corpus else []
    bm25_results = _bm25_search(query_text, docs=bm25_docs or None, top_k=top_k)

    merged: Dict[str, Dict[str, object]] = {}
    for item in bm25_results + dense_results:
        key = item["id"]
        current = merged.get(key)
        if current is None or (item.get("score") or 0) > (current.get("score") or 0):
            merged[key] = item

    ranked = sorted(
        merged.values(),
        key=lambda item: item.get("score") or 0,
        reverse=True,
    )
    if not ranked:
        fallback = [
            {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "score": 0.0,
                "source": doc["id"],
            }
            for doc in _DEFAULT_CORPUS[:top_k]
        ]
        logger.info("Hybrid search returned 0 candidates, using default corpus fallback.")
        return fallback

    logger.info("Hybrid search returned %d candidates", len(ranked))
    return ranked[:top_k]
