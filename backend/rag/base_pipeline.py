"""Base RAG pipeline interface."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from langchain_core.messages import BaseMessage

class BasePipeline(ABC):
    @abstractmethod
    def startup(self) -> None: ...
    @abstractmethod
    def respond(self, query: str, history: List[BaseMessage] = None) -> Dict[str, Any]: ...
