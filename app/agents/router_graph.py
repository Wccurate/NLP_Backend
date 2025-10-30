"""LangGraph router for directing intents to tools."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..llm_provider import ChatProvider
from ..tools.search_client import SearchProvider
from . import tools

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    intent: str
    history: str
    user_input: str
    resume_text: str
    turn_index: int
    llm: ChatProvider
    retriever: Callable[..., List[Dict[str, Any]]]
    search_client: Optional[SearchProvider]
    result: Dict[str, Any]


def _intent_node(state: AgentState) -> AgentState:
    logger.info("Routing intent: %s", state.get("intent"))
    return state


def _normal_chat_node(state: AgentState) -> AgentState:
    text, meta = tools.normal_chat(
        state["llm"],
        history=state.get("history", ""),
        user_input=state.get("user_input", ""),
        search_client=state.get("search_client"),
    )
    state["result"] = {"text": text, "meta": meta}
    return state


def _mock_interview_node(state: AgentState) -> AgentState:
    text, meta = tools.mock_interview(
        state["llm"],
        history=state.get("history", ""),
        resume_text=state.get("resume_text", ""),
        turn_index=state.get("turn_index", 0),
    )
    state["result"] = {"text": text, "meta": meta}
    return state


def _evaluate_resume_node(state: AgentState) -> AgentState:
    text, meta = tools.evaluate_resume(
        state["llm"],
        resume_text=state.get("resume_text") or state.get("user_input", ""),
    )
    state["result"] = {"text": text, "meta": meta}
    return state


def _recommend_job_node(state: AgentState) -> AgentState:
    retriever = state["retriever"]
    text, meta = tools.recommend_job(
        state["llm"], retriever=retriever, question=state.get("user_input", "")
    )
    state["result"] = {"text": text, "meta": meta}
    return state


def _compose_reply(state: AgentState) -> AgentState:
    return state


def _build_graph() -> StateGraph[AgentState]:
    graph = StateGraph(AgentState)
    graph.add_node("route_intent", _intent_node)
    graph.add_node("normal_chat", _normal_chat_node)
    graph.add_node("mock_interview", _mock_interview_node)
    graph.add_node("evaluate_resume", _evaluate_resume_node)
    graph.add_node("recommend_job", _recommend_job_node)
    graph.add_node("compose_reply", _compose_reply)

    graph.set_entry_point("route_intent")
    graph.add_conditional_edges(
        "route_intent",
        lambda state: state["intent"],
        {
            "normal_chat": "normal_chat",
            "mock_interview": "mock_interview",
            "evaluate_resume": "evaluate_resume",
            "recommend_job": "recommend_job",
        },
    )
    graph.add_edge("normal_chat", "compose_reply")
    graph.add_edge("mock_interview", "compose_reply")
    graph.add_edge("evaluate_resume", "compose_reply")
    graph.add_edge("recommend_job", "compose_reply")
    graph.add_edge("compose_reply", END)
    return graph


_GRAPH = _build_graph().compile()


def run_agent(state: AgentState) -> AgentState:
    """Execute the routing graph."""

    return _GRAPH.invoke(state)
