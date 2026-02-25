"""Session service — anonymous sessions (no user_id)."""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Any

from sqlalchemy.orm import Session as DbSession

from backend.core.models.chat_session import ChatSession
from backend.core.models.message import Message


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SessionService:
    """Manages chat sessions and messages in SQLite."""

    # ── Sessions ──────────────────────────────────────────────────────────

    def create_session(
        self, db: DbSession, title: str = "New Chat", session_id: Optional[str] = None
    ) -> ChatSession:
        session = ChatSession(id=session_id or str(uuid.uuid4()), title=title)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    def get_sessions(
        self, db: DbSession, skip: int = 0, limit: int = 50
    ) -> List[ChatSession]:
        return (
            db.query(ChatSession)
            .order_by(ChatSession.updated_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def count_sessions(self, db: DbSession) -> int:
        return db.query(ChatSession).count()

    def get_session(self, db: DbSession, session_id: str) -> Optional[ChatSession]:
        return db.query(ChatSession).filter(ChatSession.id == session_id).first()

    def update_session_title(
        self, db: DbSession, session_id: str, title: str
    ) -> Optional[ChatSession]:
        sess = self.get_session(db, session_id)
        if sess is None:
            return None
        sess.title = title
        sess.updated_at = utcnow()
        db.commit()
        db.refresh(sess)
        return sess

    def delete_session(self, db: DbSession, session_id: str) -> bool:
        sess = self.get_session(db, session_id)
        if sess is None:
            return False
        db.delete(sess)
        db.commit()
        return True

    def update_session_timestamp(self, db: DbSession, session_id: str) -> None:
        sess = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if sess:
            sess.updated_at = utcnow()
            db.commit()

    def auto_title_session(
        self, db: DbSession, session_id: str, first_message: str
    ) -> None:
        sess = self.get_session(db, session_id)
        if sess and sess.title == "New Chat":
            title = first_message[:50]
            if len(first_message) > 50:
                last_space = title.rfind(" ")
                title = title[:last_space] + "..." if last_space > 20 else title + "..."
            sess.title = title
            db.commit()

    # ── Messages ──────────────────────────────────────────────────────────

    def add_message(
        self,
        db: DbSession,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[Any] = None,
    ) -> Message:
        msg = Message(
            session_id=session_id, role=role, content=content, sources=sources
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg

    def get_messages(self, db: DbSession, session_id: str) -> List[Message]:
        return (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.timestamp)
            .all()
        )


session_service = SessionService()
