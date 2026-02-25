"""Multi-tool LangGraph agent for Vietnamese legal queries.

Tools:
  1. search_legal_database    — Qdrant vector similarity search
  2. traverse_knowledge_graph — NetworkX BFS entity traversal
  3. lookup_law_document      — Fetch full document by ID from LawDocumentService
  4. search_law_text          — BM25-style text search across documents

LangGraph flow:
  classify_query → call_tools → synthesize → END
"""

import json
import logging
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from backend.config import config

logger = logging.getLogger(__name__)


# ── Agent State ────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    tool_results: Dict[str, Any]
    final_answer: str
    sources: List[Dict]


# ── Tools ──────────────────────────────────────────────────────────────────


def _get_retriever():
    from backend.rag.langchain_pipeline import get_langchain_pipeline

    return get_langchain_pipeline().vector_store.as_retriever(search_kwargs={"k": 5})


def _get_kg():
    from backend.rag.knowledge_graph import get_knowledge_graph

    return get_knowledge_graph()


def _get_lookup():
    from backend.services.lookup_service import get_lookup_service

    return get_lookup_service()


@tool
def search_legal_database(query: str) -> str:
    """Tìm kiếm tài liệu pháp luật Việt Nam liên quan nhất bằng tìm kiếm ngữ nghĩa (vector search).
    Sử dụng khi cần tra cứu điều khoản, quy định, nghị định cụ thể."""
    try:
        retriever = _get_retriever()
        docs = retriever.invoke(query)
        if not docs:
            return "Không tìm thấy tài liệu liên quan."
        results = []
        for i, doc in enumerate(docs[:5], 1):
            meta = doc.metadata
            results.append(
                f"[{i}] {meta.get('document_title', '')} — {meta.get('article_number', '')}\n{doc.page_content[:400]}"
            )
        return "\n\n".join(results)
    except Exception as e:
        logger.error(f"search_legal_database error: {e}")
        return f"Lỗi tìm kiếm: {e}"


@tool
def traverse_knowledge_graph(entity_or_concept: str) -> str:
    """Duyệt đồ thị tri thức để tìm các văn bản pháp luật liên quan qua quan hệ (trích dẫn, sửa đổi, định nghĩa).
    Sử dụng khi cần biết văn bản nào tham chiếu hoặc sửa đổi lẫn nhau."""
    try:
        kg = _get_kg()
        doc_ids = kg.get_related_documents(entity_or_concept, hops=2)
        if not doc_ids:
            return (
                f"Không tìm thấy thực thể '{entity_or_concept}' trong đồ thị tri thức."
            )
        lookup = _get_lookup()
        summaries = []
        for doc_id in doc_ids[:5]:
            doc = lookup.get_document(doc_id)
            if doc:
                summaries.append(f"• {doc['title']}: {doc['snippet']}")
        return f"Tìm thấy {len(doc_ids)} văn bản liên quan:\n" + "\n".join(summaries)
    except Exception as e:
        logger.error(f"traverse_knowledge_graph error: {e}")
        return f"Lỗi đồ thị: {e}"


@tool
def lookup_law_document(document_id: str) -> str:
    """Lấy toàn bộ nội dung của một văn bản pháp luật theo ID hoặc số hiệu.
    Sử dụng khi cần xem chi tiết điều khoản cụ thể của một luật."""
    try:
        lookup = _get_lookup()
        doc = lookup.get_document(document_id)
        if not doc:
            return f"Không tìm thấy văn bản với ID '{document_id}'."
        articles = lookup.get_articles(document_id)
        article_summary = "\n".join(
            f"  {a['article_number']}: {a['article_title'][:80]}" for a in articles[:10]
        )
        return (
            f"**{doc['title']}**\n"
            f"Loại: {doc['type']} | Số: {doc.get('ref', '')} | Năm: {doc['date']}\n\n"
            f"Tóm tắt: {doc['snippet']}\n\n"
            f"Các điều khoản (10 đầu tiên):\n{article_summary}"
        )
    except Exception as e:
        logger.error(f"lookup_law_document error: {e}")
        return f"Lỗi tra cứu: {e}"


@tool
def search_law_text(keyword: str) -> str:
    """Tìm kiếm văn bản pháp luật theo từ khóa trong tiêu đề và nội dung (tìm kiếm văn bản).
    Sử dụng khi biết tên cụ thể của một luật cần tìm."""
    try:
        lookup = _get_lookup()
        result = lookup.search_documents(keyword, page_size=5)
        if not result["items"]:
            return f"Không tìm thấy văn bản nào với từ khóa '{keyword}'."
        lines = [f"Tìm thấy {result['total']} văn bản:"]
        for item in result["items"]:
            lines.append(f"• [{item['id']}] {item['title']} — {item['snippet'][:100]}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"search_law_text error: {e}")
        return f"Lỗi tìm kiếm: {e}"


ALL_TOOLS = [
    search_legal_database,
    traverse_knowledge_graph,
    lookup_law_document,
    search_law_text,
]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# ── Agent nodes ────────────────────────────────────────────────────────────


class MultiToolAgent:
    """LangGraph multi-tool agent for Vietnamese legal queries."""

    def __init__(self) -> None:
        self._initialized = False
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.llm_with_tools = None
        self.graph = None

    def startup(self) -> None:
        if self._initialized:
            return
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.0,
        )
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.graph = self._build_graph()
        self._initialized = True
        logger.info("✅ MultiToolAgent initialized")

    def _agent_node(self, state: AgentState) -> AgentState:
        """Call LLM with tools — decide which tools to invoke."""
        system = (
            "Bạn là trợ lý pháp lý AI chuyên nghiệp về pháp luật Việt Nam.\n"
            "Sử dụng các công cụ sau để trả lời câu hỏi một cách chính xác:\n"
            "- search_legal_database: tìm kiếm ngữ nghĩa trong Qdrant\n"
            "- traverse_knowledge_graph: duyệt đồ thị tri thức\n"
            "- lookup_law_document: lấy nội dung văn bản theo ID\n"
            "- search_law_text: tìm kiếm theo từ khóa\n"
            "Hãy sử dụng ít nhất 1 công cụ trước khi trả lời."
        )
        messages = list(state["messages"])
        if messages and messages[0].type == "human":
            content = messages[0].content
            if "HƯỚNG DẪN HỆ THỐNG:" not in content:
                messages[0] = HumanMessage(
                    content=f"HƯỚNG DẪN HỆ THỐNG:\n{system}\n\nCÂU HỎI:\n{content}"
                )
        response = self.llm_with_tools.invoke(messages)
        return {
            "messages": [response],
            "query": state["query"],
            "tool_results": state.get("tool_results", {}),
            "final_answer": state.get("final_answer", ""),
            "sources": state.get("sources", []),
        }

    def _tool_node(self, state: AgentState) -> AgentState:
        """Execute all tool calls from the last AI message."""
        last_msg = state["messages"][-1]
        tool_results = dict(state.get("tool_results", {}))
        new_messages = []
        sources = list(state.get("sources", []))

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                t = TOOL_MAP.get(tool_name)
                if t:
                    try:
                        result = t.invoke(tool_args)
                        tool_results[tool_name] = result
                        if tool_name in (
                            "search_legal_database",
                            "traverse_knowledge_graph",
                        ):
                            sources.append({"tool": tool_name, "result": result[:200]})
                    except Exception as e:
                        result = f"Tool error: {e}"
                    new_messages.append(
                        ToolMessage(content=result, tool_call_id=tc["id"])
                    )

        return {
            **state,
            "messages": new_messages,
            "tool_results": tool_results,
            "sources": sources,
        }

    def _should_continue(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", self._tool_node)
        g.set_entry_point("agent")
        g.add_conditional_edges(
            "agent", self._should_continue, {"tools": "tools", "end": END}
        )
        g.add_edge("tools", "agent")
        return g.compile()

    def respond(
        self, query: str, history: Optional[List[BaseMessage]] = None
    ) -> Dict[str, Any]:
        if not self._initialized:
            self.startup()
        messages = list(history or []) + [HumanMessage(content=query)]
        init: AgentState = {
            "messages": messages,
            "query": query,
            "tool_results": {},
            "final_answer": "",
            "sources": [],
        }
        final = self.graph.invoke(init)
        last_msg = final["messages"][-1]
        answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        return {"answer": answer, "sources": final.get("sources", [])}

    async def arespond_stream(
        self, query: str, history: Optional[List[BaseMessage]] = None
    ):
        """Async generator yielding SSE dicts."""
        import asyncio
        from functools import partial

        if not self._initialized:
            self.startup()
        loop = asyncio.get_event_loop()
        yield {"type": "status", "data": "🤖 Agent đang phân tích câu hỏi..."}
        result = await loop.run_in_executor(None, partial(self.respond, query, history))
        yield {"type": "sources", "data": result["sources"]}
        answer = result["answer"]
        for i in range(0, len(answer), 8):
            yield {"type": "chunk", "data": answer[i : i + 8]}
            await asyncio.sleep(0)
        yield {"type": "done", "data": ""}


# Singleton
_agent: Optional[MultiToolAgent] = None


def get_agent() -> MultiToolAgent:
    global _agent
    if _agent is None:
        _agent = MultiToolAgent()
        _agent.startup()
    return _agent
