"""Sessions CRUD router — anonymous sessions, no auth."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session as DbSession

from backend.core.database import get_db
from backend.services.session_service import session_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionCreate(BaseModel):
    title: Optional[str] = "New Chat"


class SessionUpdate(BaseModel):
    title: str


# ── Endpoints ──────────────────────────────────────────────────────────────


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_session(
    data: SessionCreate = SessionCreate(), db: DbSession = Depends(get_db)
):
    sess = session_service.create_session(db, title=data.title or "New Chat")
    return {
        "id": sess.id,
        "title": sess.title,
        "created_at": sess.created_at,
        "updated_at": sess.updated_at,
    }


@router.get("")
async def list_sessions(
    skip: int = 0, limit: int = 50, db: DbSession = Depends(get_db)
):
    sessions = session_service.get_sessions(db, skip=skip, limit=limit)
    total = session_service.count_sessions(db)
    return {
        "sessions": [
            {"id": s.id, "title": s.title, "updated_at": s.updated_at} for s in sessions
        ],
        "total": total,
    }


@router.get("/{session_id}")
async def get_session(session_id: str, db: DbSession = Depends(get_db)):
    sess = session_service.get_session(db, session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = session_service.get_messages(db, session_id)
    return {
        "id": sess.id,
        "title": sess.title,
        "created_at": sess.created_at,
        "updated_at": sess.updated_at,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": m.sources,
                "timestamp": m.timestamp,
            }
            for m in messages
        ],
    }


@router.patch("/{session_id}")
async def update_session(
    session_id: str, data: SessionUpdate, db: DbSession = Depends(get_db)
):
    sess = session_service.update_session_title(db, session_id, data.title)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"id": sess.id, "title": sess.title, "updated_at": sess.updated_at}


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, db: DbSession = Depends(get_db)):
    deleted = session_service.delete_session(db, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
