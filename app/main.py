"""FastAPI entrypoint for the Simple RAG + Agent backend."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, Iterable, List, Sequence

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .agents.router_graph import run_agent
from .config import get_settings
from .db import History, get_db, init_db
from .llm_provider import get_llm_provider
from .rag import vector_store
from .schemas import GenerateRequest, GenerateResponse, SourceItem
from .tools import intent as intent_tools
from .tools import web_search
from .utils import file_loader, sliding_window, text as text_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Simple RAG + Agent Demo")


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database and vector storage."""

    init_db()
    try:
        vector_store.get_collection()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector store initialization failed: %s", exc)


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""

    return {"status": "ok"}


@app.get("/history")
def get_history(
    limit: int = Query(20, gt=0, le=100),
    db: Session = Depends(get_db),
) -> List[Dict[str, str]]:
    """Return the recent conversation history."""

    stmt = select(History).order_by(History.created_at.desc()).limit(limit)
    rows = db.execute(stmt).scalars().all()
    entries = [
        {
            "role": row.role,
            "content": row.content,
            "intent": row.intent,
            "timestamp": row.created_at.isoformat(),
        }
        for row in reversed(rows)
    ]
    return entries


def _format_history(messages: Sequence[History]) -> str:
    return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)


async def _maybe_search(user_text: str, use_search: bool) -> Iterable[Dict[str, str]]:
    if not use_search or not user_text.strip():
        return []
    return await web_search.web_search(user_text)


def _count_intent(messages: Sequence[History], intent: str) -> int:
    return sum(1 for msg in messages if msg.intent == intent and msg.role == "assistant")


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    input: str = Form(""),
    web_search: bool = Form(False),
    return_stream: bool = Form(False),
    persist_documents: bool = Form(False),
    file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
) -> GenerateResponse | StreamingResponse:
    """Primary generation route handling routing and RAG."""

    request_payload = GenerateRequest(
        input=input or "",
        web_search=web_search,
        return_stream=return_stream,
        persist_documents=persist_documents,
    )

    file_text = await file_loader.extract_text_from_upload(file)

    if not request_payload.input.strip() and not file_text.strip():
        raise HTTPException(status_code=422, detail="Input text or file is required.")

    existing_docs = text_utils.extract_documents(request_payload.input)
    if file_text:
        existing_docs.append(file_text)

    user_text = text_utils.strip_document_tags(request_payload.input).strip()

    combined_input = user_text
    for doc_content in existing_docs:
        doc_content = doc_content.strip()
        if not doc_content:
            continue
        if combined_input:
            combined_input += "\n"
        combined_input += f"<document>\n{doc_content}\n</document>"

    documents_joined = "\n\n".join(existing_docs).strip()

    settings = get_settings()

    stmt = select(History).order_by(History.created_at.asc())
    past_messages = db.execute(stmt).scalars().all()
    recent_messages = sliding_window.last_k(past_messages, settings.history_window)
    history_str = _format_history(recent_messages)
    turn_index = _count_intent(past_messages, "mock_interview")

    try:
        intent = intent_tools.intent_router(user_text or combined_input)
    except Exception as exc:  # noqa: BLE001
        logger.error("Intent classification failed: %s", exc)
        raise HTTPException(status_code=500, detail="Intent classification failed.") from exc

    llm = get_llm_provider()

    search_seed = user_text or documents_joined
    search_results = await _maybe_search(search_seed, request_payload.web_search)

    chunk_ids: List[str] = []
    bm25_extra: List[Dict[str, str]] = []

    if documents_joined:
        chunks: List[str] = []
        metadatas: List[Dict[str, str]] = []
        for doc_index, doc_content in enumerate(existing_docs):
            for chunk_index, chunk in enumerate(text_utils.chunk_text(doc_content)):
                chunks.append(chunk)
                metadatas.append(
                    {
                        "source": f"user_doc#{doc_index}-{chunk_index}",
                        "type": "document",
                        "created_at": datetime.utcnow().isoformat(),
                    }
                )
        if chunks:
            try:
                chunk_ids = vector_store.add_texts(
                    chunks,
                    metadatas=metadatas,
                    embedder=llm,
                )
                bm25_extra = [
                    {"id": meta["source"], "text": chunk, "metadata": meta}
                    for chunk, meta in zip(chunks, metadatas)
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to index document: %s", exc)

    def retriever(question: str, hyde_text: str | None = None):
        return vector_store.search(
            question,
            hyde_text=hyde_text,
            extra_corpus=bm25_extra if bm25_extra else None,
            embedder=llm,
        )

    agent_state = {
        "intent": intent,
        "history": history_str,
        "user_input": user_text or combined_input,
        "resume_text": documents_joined or user_text,
        "turn_index": turn_index,
        "search_results": search_results,
        "llm": llm,
        "retriever": retriever,
    }

    try:
        agent_result = run_agent(agent_state)
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent execution failed: %s", exc)
        if chunk_ids and not request_payload.persist_documents:
            vector_store.delete(chunk_ids)
        raise HTTPException(status_code=500, detail="Generation failed.") from exc

    result_payload = agent_result.get("result", {})
    response_text = result_payload.get("text", "")
    meta = result_payload.get("meta", {})

    sources = [
        SourceItem(**item) for item in meta.get("sources", []) if isinstance(item, dict)
    ]

    tool_calls = [meta.get("tool")] if meta.get("tool") else []

    user_entry = History(role="user", content=combined_input or request_payload.input, intent=intent)
    assistant_entry = History(role="assistant", content=response_text, intent=intent)
    db.add_all([user_entry, assistant_entry])
    db.commit()

    if chunk_ids and not request_payload.persist_documents:
        vector_store.delete(chunk_ids)

    if request_payload.return_stream:
        async def stream() -> AsyncGenerator[str, None]:
            for sentence in response_text.split(". "):
                chunk = sentence.strip()
                if not chunk:
                    continue
                yield chunk + "\n"
                await asyncio.sleep(0)

        return StreamingResponse(stream(), media_type="text/plain")

    return GenerateResponse(
        intent=intent,
        text=response_text,
        sources=sources,
        tool_calls=tool_calls,
    )
