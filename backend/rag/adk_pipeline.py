"""Google ADK Streaming Pipeline — wraps Qdrant retrieval as an ADK tool.

Provides `aquery_stream()` which yields SSE-compatible dicts.
"""

import asyncio
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from backend.config import config

logger = logging.getLogger(__name__)

_adk_pipeline: Optional["ADKPipeline"] = None


class ADKPipeline:
    """Google ADK agent that calls Qdrant retrieval as a tool."""

    def __init__(self) -> None:
        self._initialized = False
        self.app_name = "raglaw_agent"

    def startup(self) -> None:
        if self._initialized:
            return
        try:
            from google.adk.agents import LlmAgent
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService
            from google.genai import types as genai_types
            from langchain_core.tools import tool

            # Build the retrieval tool
            from backend.rag.langchain_pipeline import get_langchain_pipeline

            lc_pipeline = get_langchain_pipeline()

            @tool
            def retrieve_legal_documents(query: str) -> str:
                """Tìm kiếm tài liệu pháp luật Việt Nam liên quan đến truy vấn."""
                try:
                    retriever = lc_pipeline.vector_store.as_retriever(
                        search_kwargs={"k": 5}
                    )
                    docs = retriever.invoke(query)
                    if not docs:
                        return "Không tìm thấy tài liệu liên quan."
                    return "\n\n---\n\n".join(d.page_content for d in docs)
                except Exception as e:
                    logger.error(f"Retrieval tool error: {e}")
                    return f"Lỗi tra cứu: {str(e)}"

            self._retrieve_tool = retrieve_legal_documents

            # ADK-compatible tool definition
            self._adk_tool = genai_types.FunctionDeclaration(
                name="retrieve_legal_documents",
                description="Tìm kiếm tài liệu pháp luật Việt Nam liên quan đến truy vấn.",
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "query": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description="Truy vấn pháp luật",
                        )
                    },
                    required=["query"],
                ),
            )

            system_instruction = (
                "Bạn là trợ lý pháp lý AI chuyên về pháp luật Việt Nam.\n"
                "Khi nhận câu hỏi pháp luật, SỬ DỤNG công cụ `retrieve_legal_documents` để tìm tài liệu trước khi trả lời.\n"
                "Trả lời rõ ràng, chuyên nghiệp, bằng tiếng Việt, dựa trên tài liệu pháp luật thực tế.\n"
                "Nếu câu hỏi là chào hỏi hoặc không liên quan pháp luật, trả lời thân thiện trực tiếp."
            )

            self._agent = LlmAgent(
                name="raglaw_legal_agent",
                model=config.GEMINI_MODEL,
                instruction=system_instruction,
                tools=[self._adk_tool],
            )

            class ADKCompatSessionService(InMemorySessionService):
                def get_session(self, *args, **kwargs):
                    if args and "app_name" not in kwargs:
                        kwargs["app_name"] = args[0] if len(args) > 0 else None
                        kwargs["user_id"] = args[1] if len(args) > 1 else None
                        kwargs["session_id"] = args[2] if len(args) > 2 else None
                        args = ()
                    return super().get_session(*args, **kwargs)

                async def get_session_async(self, *args, **kwargs):
                    if args and "app_name" not in kwargs:
                        kwargs["app_name"] = args[0] if len(args) > 0 else None
                        kwargs["user_id"] = args[1] if len(args) > 1 else None
                        kwargs["session_id"] = args[2] if len(args) > 2 else None
                        args = ()
                    return await super().get_session_async(*args, **kwargs)

            self._session_service = ADKCompatSessionService()
            self._runner = Runner(
                agent=self._agent,
                app_name=self.app_name,
                session_service=self._session_service,
            )
            self._initialized = True
            logger.info("✅ ADKPipeline initialized")
        except Exception as e:
            logger.error(f"ADKPipeline startup error: {e}")
            raise

    async def aquery_stream(
        self, query: str, session_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream ADK agent responses as SSE-compatible dicts."""
        if not self._initialized:
            self.startup()

        from google.genai import types as genai_types

        user_id = "anonymous"
        sid = session_id or str(uuid.uuid4())

        # Create session if not exists
        existing = await self._session_service.get_session_async(
            app_name=self.app_name, user_id=user_id, session_id=sid
        )
        if existing is None:
            await self._session_service.create_session(
                app_name=self.app_name, user_id=user_id, session_id=sid
            )

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=query)],
        )

        yield {"type": "status", "data": "🤖 ADK Agent đang xử lý..."}

        final_answer = ""
        sources: List[Dict] = []

        async for event in self._runner.run_async(
            user_id=user_id, session_id=sid, new_message=content
        ):
            if event.is_final_response():
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_answer += part.text
            elif hasattr(event, "content") and event.content:
                for part in event.content.parts:
                    if hasattr(part, "function_response") and part.function_response:
                        # Tool call happened — retrieve sources stub
                        yield {"type": "status", "data": "📚 Đang tra cứu tài liệu..."}
                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        if fc.name == "retrieve_legal_documents":
                            result = self._retrieve_tool.run(fc.args.get("query", ""))
                            sources_texts = result.split("\n\n---\n\n")
                            sources = [{"content": t[:300]} for t in sources_texts[:5]]

        yield {"type": "sources", "data": sources}

        chunk_size = 8
        for i in range(0, len(final_answer), chunk_size):
            yield {"type": "chunk", "data": final_answer[i : i + chunk_size]}
            await asyncio.sleep(0)

        yield {"type": "done", "data": ""}


def get_adk_pipeline() -> ADKPipeline:
    global _adk_pipeline
    if _adk_pipeline is None:
        _adk_pipeline = ADKPipeline()
        _adk_pipeline.startup()
    return _adk_pipeline
