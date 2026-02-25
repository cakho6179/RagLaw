"""Chat API router — 3 endpoints: sync, LangChain stream, ADK stream."""

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session as DbSession
from langchain_core.messages import HumanMessage, AIMessage

from backend.core.database import get_db
from backend.services.session_service import session_service
from backend.services.chat_service import get_chat_service, ChatService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    sources: list = []
    session_id: str


# ── Helpers ───────────────────────────────────────────────────────────────


def _ensure_session(db: DbSession, session_id: Optional[str]) -> tuple[str, bool]:
    """Get or create session, return (session_id, is_new)."""
    if not session_id:
        sess = session_service.create_session(db)
        return sess.id, True
    sess = session_service.get_session(db, session_id)
    if sess is None:
        sess = session_service.create_session(db, title="Restored Chat")
    return sess.id, False


# ── Sync endpoint ──────────────────────────────────────────────────────────


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    db: DbSession = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Synchronous chat — waits for the full answer."""
    session_id, is_new = _ensure_session(db, payload.session_id)

    # Load DB history into memory
    existing = session_service.get_messages(db, session_id)
    service.load_history_from_messages(session_id, existing)

    result = service.respond(payload.message, session_id=session_id)

    # Persist messages
    session_service.add_message(db, session_id, "user", payload.message)
    session_service.add_message(
        db, session_id, "ai", result["answer"], sources=result["sources"]
    )
    session_service.update_session_timestamp(db, session_id)
    if is_new:
        session_service.auto_title_session(db, session_id, payload.message)

    return ChatResponse(
        reply=result["answer"], sources=result["sources"], session_id=session_id
    )


# ── LangChain SSE stream ───────────────────────────────────────────────────


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    db: DbSession = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
):
    """LangChain streaming endpoint — yields SSE events."""
    session_id, is_new = _ensure_session(db, payload.session_id)
    existing = session_service.get_messages(db, session_id)
    service.load_history_from_messages(session_id, existing)

    pipeline = service.pipeline

    async def event_generator():
        full_answer = ""
        all_sources = []
        async for event in pipeline.query_stream(
            payload.message,
            history=service._get_history(session_id).messages,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") == "chunk":
                full_answer += event.get("data", "")
            elif event.get("type") == "sources":
                all_sources = event.get("data", [])

        # Persist after streaming
        session_service.add_message(db, session_id, "user", payload.message)
        session_service.add_message(
            db, session_id, "ai", full_answer, sources=all_sources
        )
        session_service.update_session_timestamp(db, session_id)
        if is_new:
            session_service.auto_title_session(db, session_id, payload.message)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ── ADK SSE stream ────────────────────────────────────────────────────────


@router.post("/adk/stream")
async def adk_chat_stream(
    payload: ChatRequest,
    db: DbSession = Depends(get_db),
):
    """Google ADK agent streaming endpoint."""
    session_id, is_new = _ensure_session(db, payload.session_id)

    try:
        from backend.rag.adk_pipeline import get_adk_pipeline

        adk = get_adk_pipeline()
    except Exception as e:
        logger.error(f"ADK init error: {e}")

        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

        return StreamingResponse(error_gen(), media_type="text/event-stream")

    async def event_generator():
        full_answer = ""
        all_sources = []
        async for event in adk.aquery_stream(payload.message, session_id=session_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") == "chunk":
                full_answer += event.get("data", "")
            elif event.get("type") == "sources":
                all_sources = event.get("data", [])

        session_service.add_message(db, session_id, "user", payload.message)
        session_service.add_message(
            db, session_id, "ai", full_answer, sources=all_sources
        )
        session_service.update_session_timestamp(db, session_id)
        if is_new:
            session_service.auto_title_session(db, session_id, payload.message)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
