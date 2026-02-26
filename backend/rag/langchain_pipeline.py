"""LangGraph Agentic RAG Pipeline for Vietnamese Legal Documents.

Workflow:
  route_query → [retrieve → grade_documents → rewrite (×3) OR generate_direct] → generate

Yields SSE-compatible dicts for streaming.
"""

import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient

from backend.config import config
from backend.rag.adapters import (
    VietnameseEmbedding,
    VietnameseDenseAdapter,
    VietnameseSparseAdapter,
)

logger = logging.getLogger(__name__)

MAX_REWRITE = 3

_RETRIEVAL_MODE_MAP = {
    "hybrid": RetrievalMode.HYBRID,
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
}


class AgenticState(TypedDict):
    query: str
    current_query: str
    history: List[BaseMessage]
    documents: List[Document]
    relevant_documents: List[Document]
    rewrite_count: int
    needs_retrieval: bool
    answer: str
    sources: List[Dict[str, Any]]
    model_name: str


class AgenticRAGPipeline:
    """
    LangGraph agentic RAG pipeline:
      1. Route query — decide if retrieval is needed
      2. Retrieve from Qdrant
      3. Grade relevance
      4. Rewrite query up to MAX_REWRITE times
      5. Generate answer
    """

    def __init__(
        self,
        retrieval_mode: str = "dense",
        use_reranker: bool = False,
        collection_name: Optional[str] = None,
        retrieval_k: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.retrieval_mode = retrieval_mode
        self.use_reranker = use_reranker
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.retrieval_k = retrieval_k or config.RETRIEVAL_K
        self.top_k = top_k or config.TOP_K
        self._initialized = False
        self.vector_store: Optional[QdrantVectorStore] = None
        self.model: Optional[ChatGoogleGenerativeAI] = None
        self.graph = None

    # ── Init ──────────────────────────────────────────────────────────────

    def startup(self) -> None:
        if self._initialized:
            return
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        vn_emb = VietnameseEmbedding(
            model_name=config.EMBEDDING_MODEL, device=config.DEVICE
        )
        dense_adapter = VietnameseDenseAdapter(vn_emb)
        sparse_adapter = VietnameseSparseAdapter(vn_emb)

        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        mode = _RETRIEVAL_MODE_MAP.get(self.retrieval_mode, RetrievalMode.DENSE)

        if mode == RetrievalMode.HYBRID:
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
                embedding=dense_adapter,
                sparse_embedding=sparse_adapter,
                retrieval_mode=mode,
                content_payload_key="text",
            )
        else:
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
                embedding=dense_adapter,
                retrieval_mode=mode,
                content_payload_key="text",
            )

        self.models = {
            "flash": ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.0,
            ),
            "pro": ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.0,
            ),
            "gemma": ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.0,
            ),
        }
        try:
            from langchain_community.chat_models import ChatOllama

            self.models["local"] = ChatOllama(model="llama3", temperature=0.0)
        except ImportError:
            self.models["local"] = self.models["gemma"]

        self.model = self.models["gemma"]  # Default fallback

        self.graph = self._build_graph()
        self._initialized = True
        logger.info("✅ AgenticRAGPipeline initialized")

    # ── Graph nodes ───────────────────────────────────────────────────────

    def _contextualize_query(self, state: AgenticState) -> AgenticState:
        """Rewrite the query using chat history for standalone context."""
        history = state.get("history", [])
        if not history:
            # No previous conversation, use query as-is
            state["current_query"] = state["query"]
            return state

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("history"),
                (
                    "human",
                    (
                        "Dựa trên lịch sử hội thoại ở trên và câu hỏi mới nhất của người dùng, "
                        "hãy viết lại câu hỏi thành một câu độc lập, đầy đủ ngữ cảnh để có thể hiểu được mà không cần đọc lại lịch sử.\n"
                        "Nếu câu hỏi đã đầy đủ, giữ nguyên. Chỉ trả về câu hỏi đã viết lại, không giải thích gì thêm.\n\n"
                        "Câu hỏi: {query}"
                    ),
                ),
            ]
        )
        llm = self.models.get(state.get("model_name", "gemma"), self.model)
        chain = contextualize_prompt | llm | StrOutputParser()
        rewritten = chain.invoke({"history": history, "query": state["query"]}).strip()
        logger.info(f"Query contextualized: '{state['query']}' -> '{rewritten}'")
        state["current_query"] = rewritten
        return state

    def _route_query(self, state: AgenticState) -> AgenticState:
        """Decide if the query needs legal retrieval."""
        router_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    (
                        "Bạn là bộ phân loại câu hỏi cho hệ thống hỏi đáp pháp luật Việt Nam.\n"
                        "Trả lời 'yes' nếu câu hỏi liên quan đến pháp luật, quy định, điều khoản, nghị định, thông tư, bộ luật.\n"
                        "Trả lời 'no' nếu câu hỏi là câu chào hỏi, tán gẫu, hoặc không liên quan đến pháp luật."
                    ),
                ),
                ("human", "{query}"),
            ]
        )
        llm = self.models.get(state.get("model_name", "gemma"), self.model)
        chain = router_prompt | llm | StrOutputParser()
        # Use the contextualized query for better routing
        result = chain.invoke({"query": state["current_query"]}).strip().lower()
        state["needs_retrieval"] = "yes" in result
        return state

    def _retrieve(self, state: AgenticState) -> AgenticState:
        """Retrieve documents from Qdrant."""
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.retrieval_k}
        )
        docs = retriever.invoke(state["current_query"])
        state["documents"] = docs
        return state

    def _generate_with_context(self, state: AgenticState) -> AgenticState:
        """Generate answer using retrieved context."""
        formatted_context_parts = []
        for i, doc in enumerate(state["documents"][: self.top_k]):
            title = doc.metadata.get("document_title", "Tài liệu không tên")
            formatted_context_parts.append(
                f"Tài liệu [{i + 1}] ({title}):\n{doc.page_content}"
            )

        context = "\n\n---\n\n".join(formatted_context_parts)

        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("history"),
                (
                    "human",
                    (
                        "Bạn là trợ lý pháp lý chuyên nghiệp về pháp luật Việt Nam của hệ thống RagLaw.\n"
                        "Hãy trả lời câu hỏi DỰA TRÊN các TÀI LIỆU được cung cấp dưới đây.\n"
                        "BẮT BUỘC phải trích dẫn nguồn bằng cách thêm số thứ tự tài liệu trong ngoặc vuông (ví dụ: [1], [2]) vào cuối mỗi câu hoặc đoạn văn mà bạn lấy thông tin từ tài liệu đó.\n"
                        "Nếu thông tin không đủ để trả lời, hãy nói rõ thay vì tự bịa đặt.\n"
                        "Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu và chuyên nghiệp.\n\n"
                        "TÀI LIỆU:\n{context}\n\n"
                        "CÂU HỎI: {query}"
                    ),
                ),
            ]
        )
        llm = self.models.get(state.get("model_name", "gemma"), self.model)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke(
            {
                "context": context,
                "history": state.get("history", []),
                "query": state["query"],
            }
        )
        state["answer"] = answer
        state["sources"] = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in state["documents"][: self.top_k]
        ]
        return state

    def _generate_direct(self, state: AgenticState) -> AgenticState:
        """Generate answer directly (no retrieval needed)."""
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("history"),
                (
                    "human",
                    (
                        "Bạn là trợ lý pháp lý thân thiện của hệ thống RagLaw. Hãy trả lời lịch sự bằng tiếng Việt.\n\n"
                        "CÂU HỎI: {query}"
                    ),
                ),
            ]
        )
        llm = self.models.get(state.get("model_name", "gemma"), self.model)
        chain = prompt | llm | StrOutputParser()
        state["answer"] = chain.invoke(
            {"history": state.get("history", []), "query": state["query"]}
        )
        state["sources"] = []
        return state

    # ── Routing conditions ────────────────────────────────────────────────

    def _should_retrieve(self, state: AgenticState) -> str:
        return "retrieve" if state["needs_retrieval"] else "generate_direct"

    # ── Build graph ───────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(AgenticState)

        g.add_node("contextualize", self._contextualize_query)
        g.add_node("route", self._route_query)
        g.add_node("retrieve", self._retrieve)
        g.add_node("generate", self._generate_with_context)
        g.add_node("generate_direct", self._generate_direct)

        g.set_entry_point("contextualize")
        g.add_edge("contextualize", "route")
        g.add_conditional_edges("route", self._should_retrieve)
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        g.add_edge("generate_direct", END)

        return g.compile()

    # ── Public interface ──────────────────────────────────────────────────

    def respond(
        self,
        query: str,
        history: List[BaseMessage] | None = None,
        model_name: str = "gemma",
    ) -> Dict[str, Any]:
        if not self._initialized:
            self.startup()
        initial: AgenticState = {
            "query": query,
            "current_query": query,
            "history": history or [],
            "documents": [],
            "relevant_documents": [],
            "rewrite_count": 0,
            "needs_retrieval": True,
            "answer": "",
            "sources": [],
            "model_name": model_name,
        }
        final_state = self.graph.invoke(initial)
        return {"answer": final_state["answer"], "sources": final_state["sources"]}

    async def query_stream(
        self,
        query: str,
        history: List[BaseMessage] | None = None,
        model_name: str = "gemma",
    ) -> AsyncGenerator[Dict, None]:
        """Run the pipeline and yield SSE-compatible dicts with thoughts."""
        import asyncio

        if not self._initialized:
            self.startup()

        initial: AgenticState = {
            "query": query,
            "current_query": query,
            "history": history or [],
            "documents": [],
            "relevant_documents": [],
            "rewrite_count": 0,
            "needs_retrieval": True,
            "answer": "",
            "sources": [],
            "model_name": model_name,
        }

        yield {"type": "status", "data": "🔍 Đang phân tích câu hỏi..."}

        # Use LangGraph's async stream to yield thoughts at each node
        async for step in self.graph.astream(initial, stream_mode="updates"):
            for node_name, state_update in step.items():
                if node_name == "contextualize":
                    cq = state_update.get("current_query", query)
                    if cq != query:
                        yield {
                            "type": "thought",
                            "data": {
                                "agent": "Contextualizer",
                                "text": f"Viết lại câu hỏi: '{cq}'",
                            },
                        }
                    else:
                        yield {
                            "type": "thought",
                            "data": {
                                "agent": "Contextualizer",
                                "text": "Câu hỏi đã đầy đủ ngữ cảnh.",
                            },
                        }
                elif node_name == "route":
                    needs_retrieval = state_update.get("needs_retrieval", True)
                    decision = (
                        "Cần phân tích tài liệu pháp luật"
                        if needs_retrieval
                        else "Giao tiếp thông thường, bỏ qua truy xuất"
                    )
                    yield {
                        "type": "thought",
                        "data": {"agent": "Router", "text": decision},
                    }
                elif node_name == "retrieve":
                    docs = state_update.get("documents", [])
                    yield {
                        "type": "thought",
                        "data": {
                            "agent": "Retriever",
                            "text": f"Đã trích xuất {len(docs)} tài liệu liên quan từ CSDL.",
                        },
                    }
                elif node_name == "generate" or node_name == "generate_direct":
                    yield {
                        "type": "thought",
                        "data": {
                            "agent": "Generator",
                            "text": "Đang tổng hợp thông tin và phản hồi...",
                        },
                    }

                    if "sources" in state_update and state_update["sources"]:
                        yield {"type": "sources", "data": state_update["sources"]}

                    answer = state_update.get("answer", "")
                    # Stream tokens out linearly for the UX effect
                    chunk_size = 8
                    for i in range(0, len(answer), chunk_size):
                        yield {"type": "chunk", "data": answer[i : i + chunk_size]}
                        await asyncio.sleep(0.01)

        yield {"type": "done", "data": ""}


# Singleton
_pipeline: Optional[AgenticRAGPipeline] = None


def get_langchain_pipeline() -> AgenticRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AgenticRAGPipeline(retrieval_mode=config.RETRIEVAL_MODE)
        _pipeline.startup()
    return _pipeline


def reset_langchain_pipeline() -> None:
    global _pipeline
    _pipeline = None
