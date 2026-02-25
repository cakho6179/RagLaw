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
            )
        else:
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
                embedding=dense_adapter,
                retrieval_mode=mode,
            )

        self.model = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.0,
        )

        self.graph = self._build_graph()
        self._initialized = True
        logger.info("✅ AgenticRAGPipeline initialized")

    # ── Graph nodes ───────────────────────────────────────────────────────

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
        chain = router_prompt | self.model | StrOutputParser()
        result = chain.invoke({"query": state["query"]}).strip().lower()
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

    def _grade_documents(self, state: AgenticState) -> AgenticState:
        """Grade each document for relevance to query."""
        grader_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    (
                        "Bạn là bộ chấm điểm tài liệu.\n"
                        "Trả lời 'yes' nếu đoạn văn bản dưới đây có liên quan và hữu ích để trả lời câu hỏi.\n"
                        "Trả lời 'no' nếu không liên quan."
                    ),
                ),
                ("human", "Câu hỏi: {query}\n\nTài liệu:\n{document}"),
            ]
        )
        chain = grader_prompt | self.model | StrOutputParser()
        relevant = []
        for doc in state["documents"]:
            score = chain.invoke(
                {"query": state["current_query"], "document": doc.page_content}
            )
            if "yes" in score.lower():
                relevant.append(doc)
        state["relevant_documents"] = relevant
        return state

    def _rewrite_query(self, state: AgenticState) -> AgenticState:
        """Rewrite the query for better retrieval."""
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    "Bạn là chuyên gia viết lại câu truy vấn pháp luật cho tìm kiếm ngữ nghĩa. Viết lại câu hỏi súc tích hơn, đúng thuật ngữ pháp luật.",
                ),
                ("human", "Câu hỏi gốc: {query}\n\nCâu hỏi viết lại:"),
            ]
        )
        chain = rewrite_prompt | self.model | StrOutputParser()
        state["current_query"] = chain.invoke({"query": state["current_query"]}).strip()
        state["rewrite_count"] += 1
        return state

    def _generate_with_context(self, state: AgenticState) -> AgenticState:
        """Generate answer using retrieved context."""
        context = "\n\n---\n\n".join(
            doc.page_content for doc in state["relevant_documents"]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    (
                        "Bạn là trợ lý pháp lý chuyên nghiệp về pháp luật Việt Nam.\n"
                        "Hãy trả lời câu hỏi DỰA TRÊN các tài liệu pháp luật được cung cấp.\n"
                        "Nếu thông tin không đủ, hãy nói rõ thay vì tự bịa đặt.\n"
                        "Trả lời bằng tiếng Việt, rõ ràng và chuyên nghiệp.\n\n"
                        "TÀI LIỆU:\n{context}"
                    ),
                ),
                MessagesPlaceholder("history"),
                ("human", "{query}"),
            ]
        )
        chain = prompt | self.model | StrOutputParser()
        answer = chain.invoke(
            {
                "context": context,
                "history": state.get("history", []),
                "query": state["query"],
            }
        )
        state["answer"] = answer
        state["sources"] = [
            {"content": doc.page_content[:300], "metadata": doc.metadata}
            for doc in state["relevant_documents"][: self.top_k]
        ]
        return state

    def _generate_direct(self, state: AgenticState) -> AgenticState:
        """Generate answer directly (no retrieval needed)."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    "Bạn là trợ lý pháp lý thân thiện của hệ thống RagLaw. Hãy trả lời lịch sự bằng tiếng Việt.",
                ),
                MessagesPlaceholder("history"),
                ("human", "{query}"),
            ]
        )
        chain = prompt | self.model | StrOutputParser()
        state["answer"] = chain.invoke(
            {"history": state.get("history", []), "query": state["query"]}
        )
        state["sources"] = []
        return state

    # ── Routing conditions ────────────────────────────────────────────────

    def _should_retrieve(self, state: AgenticState) -> str:
        return "retrieve" if state["needs_retrieval"] else "generate_direct"

    def _should_rewrite_or_generate(self, state: AgenticState) -> str:
        if state["relevant_documents"]:
            return "generate"
        if state["rewrite_count"] < MAX_REWRITE:
            return "rewrite"
        # Give up — generate with whatever we have
        state["relevant_documents"] = state["documents"][: self.top_k]
        return "generate"

    # ── Build graph ───────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(AgenticState)

        g.add_node("route", self._route_query)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade_documents)
        g.add_node("rewrite", self._rewrite_query)
        g.add_node("generate", self._generate_with_context)
        g.add_node("generate_direct", self._generate_direct)

        g.set_entry_point("route")
        g.add_conditional_edges("route", self._should_retrieve)
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade",
            self._should_rewrite_or_generate,
            {
                "rewrite": "rewrite",
                "generate": "generate",
            },
        )
        g.add_edge("rewrite", "retrieve")
        g.add_edge("generate", END)
        g.add_edge("generate_direct", END)

        return g.compile()

    # ── Public interface ──────────────────────────────────────────────────

    def respond(
        self, query: str, history: List[BaseMessage] | None = None
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
        }
        final_state = self.graph.invoke(initial)
        return {"answer": final_state["answer"], "sources": final_state["sources"]}

    async def query_stream(
        self, query: str, history: List[BaseMessage] | None = None
    ) -> AsyncGenerator[Dict, None]:
        """Run the pipeline and yield SSE-compatible dicts."""
        import asyncio
        from functools import partial

        loop = asyncio.get_event_loop()

        yield {"type": "status", "data": "🔍 Đang phân tích câu hỏi..."}

        # Run the blocking graph in a thread
        result = await loop.run_in_executor(None, partial(self.respond, query, history))

        # Yield sources first
        yield {"type": "sources", "data": result["sources"]}

        # Stream the answer token by token (simulate since LangGraph is sync here)
        answer = result["answer"]
        chunk_size = 8
        for i in range(0, len(answer), chunk_size):
            yield {"type": "chunk", "data": answer[i : i + chunk_size]}
            await asyncio.sleep(0)

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
