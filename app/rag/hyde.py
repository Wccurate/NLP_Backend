"""HyDE helper for generating hypothetical answers."""

from __future__ import annotations

from ..llm_provider import ChatProvider, get_llm_provider


def hyde_query(question: str, llm: ChatProvider | None = None) -> str:
    """Produce a concise hypothetical answer for better retrieval."""

    llm = llm or get_llm_provider()
    prompt = (
        "Craft a concise, ideal answer to the question below. "
        "Keep it under 80 words.\n\n"
        f"Question: {question}"
    )
    return llm.chat(prompt).strip()

