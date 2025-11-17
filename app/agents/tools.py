"""Agent tool functions for different conversation intents."""

from __future__ import annotations

import logging
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..llm_provider import ChatProvider
from ..rag.hyde import hyde_query
from ..tools.search_client import SearchProvider

logger = logging.getLogger(__name__)


def normal_chat(
    llm: ChatProvider,
    *,
    history: str,
    user_input: str,
    search_client: Optional[SearchProvider] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Hold a general conversation, optionally enriched by MCP-style web search."""

    plan = {
        "search": False,
        "queries": [],
        "reason": "",
    }
    search_blocks: List[str] = []
    executed_queries: List[str] = []

    if search_client is not None:
        planner_prompt = (
            "You operate a Model Context Protocol (MCP) search tool. Using the"
            " latest user message and recent history, decide if web search is"
            " necessary. Respond with JSON only: {\"search\": bool,"
            " \"queries\": [list of focused keyword queries not phrased as"
            " direct questions], \"reason\": string}. If no search is needed,"
            " return search=false."
            "\n\nRecent conversation:\n"
            f"{history}\n\nUser: {user_input}\n"
        )
        try:
            raw_plan = llm.chat(planner_prompt).strip()
            plan = json.loads(raw_plan)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Search planning failed (%s); continuing without search.", exc)
            plan = {"search": False, "queries": []}

        if isinstance(plan, dict) and plan.get("search") and plan.get("queries"):
            queries = [str(q).strip() for q in plan.get("queries", []) if str(q).strip()]
            for query in queries[:2]:
                results = search_client.search(query, max_results=4)
                if not results:
                    continue
                executed_queries.append(query)
                snippet_lines = [
                    f"- {hit.get('title','Result')}: {hit.get('snippet','')[:280]}"
                    for hit in results
                ]
                block = (
                    f"Search query: {query}\n" + "\n".join(snippet_lines[:4])
                )
                search_blocks.append(block)

    search_context = ""
    if search_blocks:
        search_context = "External findings:\n" + "\n\n".join(search_blocks) + "\n\n"

    prompt = (
        "You are a career assistant carrying on a friendly conversation."
        " Reference external findings if they are relevant.\n"
        f"{search_context}"
        "Recent conversation:\n"
        f"{history}\n\n"
        f"User: {user_input}\n"
        "Assistant:"
    )
    response = llm.chat(prompt).strip()
    meta: Dict[str, Any] = {"tool": "normal_chat"}
    if executed_queries:
        meta["web_search"] = {"queries": executed_queries, "plan": plan.get("reason", "")}
    return response, meta


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
    """Evaluate the resume and return a Markdown formatted analysis."""

    prompt = (
        "You are a professional resume reviewer. Your task is to analyze the provided resume and generate a comprehensive evaluation in Markdown format.\n\n"
        "The evaluation must include the following sections:\n"
        "1.  **Resume Highlights**: A concise summary of the candidate's most impressive qualifications and experiences.\n"
        "2.  **Strengths**: A bulleted list of the resume's strong points.\n"
        "3.  **Areas for Improvement**: A bulleted list of weaknesses or areas that could be improved.\n"
        "4.  **Suggestions**: Actionable advice for enhancing the resume.\n\n"
        "Format your response strictly in Markdown.\n\n"
        f"Resume:\n{resume_text}\n\n"
        "Begin your evaluation now."
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
            "score": float(item.get("hybrid_score") or item.get("score") or 0.0),
            "text": item.get("text", ""),
            "dense_score": item.get("dense_score"),
            "bm25_score": item.get("bm25_score"),
            "hybrid_score": item.get("hybrid_score"),
            "bm25_raw_score": item.get("bm25_raw_score"),
            "dense_distance": item.get("dense_distance"),
        }
        for idx, item in enumerate(results)
    ]
    return answer, {"tool": "recommend_job", "sources": sources}
