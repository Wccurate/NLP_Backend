"""FastAPI entrypoint for the Simple RAG + Agent backend."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Sequence

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .agents.router_graph import run_agent
from .config import get_settings
from .db import History, get_db, init_db
from .llm_provider import get_llm_provider
from .rag import vector_store
from .schemas import (
    GenerateRequest,
    GenerateResponse,
    JobDescriptionRequest,
    JobDescriptionResponse,
    SourceItem,
)
from .tools import intent as intent_tools
from .tools.search_client import get_search_client
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

    doc_texts: List[str] = []
    if file_text:
        doc_texts.append(file_text.strip())

    user_text = request_payload.input.strip()

    if not user_text and not doc_texts:
        raise HTTPException(status_code=422, detail="Input text or file is required.")

    combined_input = user_text
    for doc_content in doc_texts:
        if combined_input:
            combined_input += "\n"
        combined_input += f"<document>\n{doc_content}\n</document>"

    documents_joined = "\n\n".join(doc_texts).strip()

    settings = get_settings()

    stmt = select(History).order_by(History.created_at.asc())
    past_messages = db.execute(stmt).scalars().all()
    recent_messages = sliding_window.last_k(past_messages, settings.history_window)
    history_str = _format_history(recent_messages)
    turn_index = _count_intent(past_messages, "mock_interview")

    try:
        intent = intent_tools.intent_router(user_text or documents_joined or combined_input)
    except Exception as exc:  # noqa: BLE001
        logger.error("Intent classification failed: %s", exc)
        raise HTTPException(status_code=500, detail="Intent classification failed.") from exc

    llm = get_llm_provider()
    search_client = get_search_client()

    def retriever(question: str, hyde_text: str | None = None):
        return vector_store.search(question, hyde_text=hyde_text, embedder=llm)

    agent_state = {
        "intent": intent,
        "history": history_str,
        "user_input": user_text or documents_joined or combined_input,
        "resume_text": documents_joined or user_text,
        "turn_index": turn_index,
        "llm": llm,
        "retriever": retriever,
        "search_client": search_client,
    }

    try:
        agent_result = run_agent(agent_state)
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent execution failed: %s", exc)
        raise HTTPException(status_code=500, detail="Generation failed.") from exc

    result_payload = agent_result.get("result", {})
    response_text = result_payload.get("text", "")
    meta = result_payload.get("meta", {})

    sources = [
        SourceItem(**item) for item in meta.get("sources", []) if isinstance(item, dict)
    ]

    tool_calls = [meta.get("tool")] if meta.get("tool") else []

    user_entry = History(role="user", content=combined_input or documents_joined, intent=intent)
    assistant_entry = History(role="assistant", content=response_text, intent=intent)
    db.add_all([user_entry, assistant_entry])
    db.commit()

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
@app.post("/jobs", response_model=JobDescriptionResponse, status_code=201)
def add_job_description(payload: JobDescriptionRequest) -> JobDescriptionResponse:
    """Persist a job description into the vector store for RAG."""

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Job description text is required.")

    chunks = text_utils.chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Job description text is empty.")

    llm = get_llm_provider()
    timestamp = datetime.utcnow().isoformat()
    base_source = payload.title or "job_description"
    extra_meta = payload.metadata or {}

    metadatas = []
    for idx, _ in enumerate(chunks):
        meta = {
            "source": f"{base_source}#{idx}",
            "title": payload.title,
            "type": "job_description",
            "created_at": timestamp,
        }
        meta.update(extra_meta)
        metadatas.append(meta)

    try:
        ids = vector_store.add_texts(chunks, metadatas=metadatas, embedder=llm)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to index job description: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to index job description.") from exc

    logger.info("Indexed job description with %d chunks", len(ids))
    return JobDescriptionResponse(inserted=len(ids), ids=ids)

