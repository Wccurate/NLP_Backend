"""Agent tool functions for different conversation intents."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..llm_provider import ChatProvider
from ..rag.hyde import hyde_query

logger = logging.getLogger(__name__)


def normal_chat(
    llm: ChatProvider,
    *,
    history: str,
    user_input: str,
    search_results: Optional[Iterable[Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Hold a general conversation, optionally enriched by web search."""

    search_context = ""
    if search_results:
        snippets = [
            f"- {item.get('title','Unknown')}: {item.get('snippet','')}"[:300]
            for item in search_results
        ]
        if snippets:
            search_context = "Web findings:\n" + "\n".join(snippets) + "\n\n"

    prompt = (
        "You are a career assistant carrying on a friendly conversation.\n"
        f"{search_context}"
        "Recent conversation:\n"
        f"{history}\n\n"
        f"User: {user_input}\n"
        "Assistant:"
    )
    response = llm.chat(prompt).strip()
    return response, {"tool": "normal_chat"}


def mock_interview(
    llm: ChatProvider,
    *,
    history: str,
    resume_text: str = "",
    turn_index: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Generate one interview question and rubric."""

    prompt = (
        "You are conducting a mock interview. Provide exactly one interview question "
        "followed by a short scoring rubric.\n"
        "Increase difficulty as the session progresses. Use the turn index to gauge "
        "difficulty: 0 is easy, 1-2 moderate, >2 advanced.\n"
        "Format:\n"
        "Question: ...\n"
        "Rubric:\n"
        "- ...\n"
        "- ...\n"
        "- ...\n"
        "Resume context (if any):\n"
        f"{resume_text}\n\n"
        "Recent conversation:\n"
        f"{history}\n\n"
        f"Turn index: {turn_index}\n"
        "Provide the response now."
    )
    response = llm.chat(prompt).strip()
    return response, {"tool": "mock_interview", "turn_index": turn_index}


def evaluate_resume(
    llm: ChatProvider,
    *,
    resume_text: str,
) -> Tuple[str, Dict[str, Any]]:
    """Evaluate the resume and return a JSON string."""

    prompt = (
        "You are a resume reviewer. Read the resume text and respond with JSON only.\n"
        'Keys: "pros" (array of strengths), "cons" (array of issues), '
        '"suggestions" (array of concise advice).\n'
        "Resume:\n"
        f"{resume_text}\n"
        "JSON:"
    )
    response = llm.chat(prompt).strip()
    return response, {"tool": "evaluate_resume"}


def recommend_job(
    llm: ChatProvider,
    *,
    retriever: Callable[..., List[Dict[str, Any]]],
    question: str,
) -> Tuple[str, Dict[str, Any]]:
    """Recommend jobs using hybrid retrieval with HyDE prompting."""

    synthetic = hyde_query(question, llm=llm)
    results = retriever(question, hyde_text=synthetic)

    if not results:
        fallback = (
            "I could not find matching roles right now. "
            "Consider refining your interests or sharing more details."
        )
        return fallback, {"tool": "recommend_job", "sources": []}

    context = "\n\n".join(
        f"[{idx}] {item.get('text','')}" for idx, item in enumerate(results)
    )
    prompt = (
        "You are a job matching assistant. Use the provided context to recommend roles. "
        "Cite supporting snippets with bracketed numbers like [0].\n"
        f"User question: {question}\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
    answer = llm.chat(prompt).strip()
    sources = [
        {
            "source": item.get("source") or item.get("id") or f"jobs_demo#{idx}",
            "score": float(item.get("score") or 0.0),
            "text": item.get("text", ""),
        }
        for idx, item in enumerate(results)
    ]
    return answer, {"tool": "recommend_job", "sources": sources}

