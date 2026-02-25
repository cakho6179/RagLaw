"""Chat service wrapping the AgenticRAGPipeline with in-memory session history."""

import logging
from typing import Dict, Any, Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from backend.rag.langchain_pipeline import AgenticRAGPipeline, get_langchain_pipeline

logger = logging.getLogger(__name__)


class ChatService:
    """Orchestrates RAGPipeline with per-session in-memory chat history."""

    def __init__(self, pipeline: Optional[AgenticRAGPipeline] = None) -> None:
        self.pipeline = pipeline or get_langchain_pipeline()
        self.session_store: Dict[str, BaseChatMessageHistory] = {}

    def startup(self) -> None:
        self.pipeline.startup()

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]

    def load_history_from_messages(self, session_id: str, messages) -> None:
        """Populate in-memory history from DB messages (after server restart)."""
        history = ChatMessageHistory()
        for msg in messages:
            if msg.role == "user":
                history.add_message(HumanMessage(content=msg.content))
            elif msg.role == "ai":
                history.add_message(AIMessage(content=msg.content))
        self.session_store[session_id] = history

    def respond(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        history = self._get_history(session_id)
        result = self.pipeline.respond(query, history=history.messages)
        history.add_user_message(query)
        history.add_ai_message(result["answer"])
        return result


# Singleton
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
        _chat_service.startup()
    return _chat_service
