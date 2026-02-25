"""GraphRAG Pipeline — hybrid vector search + Knowledge Graph traversal with RRF fusion.

Provides:
  GraphRAGPipeline.respond()       — sync dict result
  GraphRAGPipeline.graph_stream()  — async SSE generator
"""

import asyncio
import logging
from functools import partial
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from backend.config import config
from backend.rag.adapters import VietnameseEmbedding, VietnameseDenseAdapter
from backend.rag.knowledge_graph import KnowledgeGraph, get_knowledge_graph
from backend.services.lookup_service import LawDocumentService, get_lookup_service

logger = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    results_lists: List[List[Document]], k: int = 60
) -> List[Document]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content[:120]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    return [doc_map[k] for k in sorted_keys]


class GraphRAGPipeline:
    """
    GraphRAG = Qdrant vector retrieval ⊕ Knowledge Graph traversal + RRF fusion.

    1. Extract query entities from KG → get related doc_ids
    2. Filter Qdrant results to prioritise KG-related docs
    3. Merge with plain vector results via RRF
    4. Generate using top-K fused context
    """

    def __init__(
        self,
        retrieval_k: int = 10,
        top_k: int = 5,
        kg_hops: int = 2,
    ) -> None:
        self.retrieval_k = retrieval_k
        self.top_k = top_k
        self.kg_hops = kg_hops
        self._initialized = False
        self.vector_store: Optional[QdrantVectorStore] = None
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.kg: Optional[KnowledgeGraph] = None
        self.lookup: Optional[LawDocumentService] = None

    def startup(self) -> None:
        if self._initialized:
            return
        logger.info("GraphRAGPipeline: initializing...")
        vn_emb = VietnameseEmbedding(
            model_name=config.EMBEDDING_MODEL, device=config.DEVICE
        )
        adapter = VietnameseDenseAdapter(vn_emb)
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embedding=adapter,
            retrieval_mode=RetrievalMode.DENSE,
            content_payload_key="text",
        )
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.0,
        )
        self.kg = get_knowledge_graph()
        self.lookup = get_lookup_service()
        self._initialized = True
        logger.info("✅ GraphRAGPipeline initialized")

    # ── Core retrieval ─────────────────────────────────────────────────────

    def _retrieve_vector(self, query: str) -> List[Document]:
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.retrieval_k}
        )
        return retriever.invoke(query)

    def _retrieve_graph(self, query: str) -> List[Document]:
        """Traverse KG to find related doc_ids, then fetch their Qdrant chunks."""
        doc_ids = self.kg.get_related_documents(query, hops=self.kg_hops)
        if not doc_ids:
            return []
        # Filter vector store results by matching doc_ids in metadata
        all_results: List[Document] = []
        for doc_id in doc_ids[:5]:
            try:
                results = self.vector_store.similarity_search(
                    query, k=3, filter={"document_id": doc_id}
                )
                all_results.extend(results)
            except Exception:
                pass
        return all_results

    def _fuse_and_rerank(
        self, vector_docs: List[Document], graph_docs: List[Document]
    ) -> List[Document]:
        """Merge vector + graph results via RRF, return top_k."""
        fused = _reciprocal_rank_fusion([vector_docs, graph_docs])
        return fused[: self.top_k]

    def _generate(
        self, query: str, docs: List[Document], history: List[BaseMessage]
    ) -> str:
        context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else ""
        system = (
            "Bạn là trợ lý pháp lý AI chuyên về pháp luật Việt Nam.\n"
            "Hãy trả lời câu hỏi DỰA TRÊN tài liệu pháp luật được cung cấp.\n"
            f"{'TÀI LIỆU PHÁP LUẬT (từ cả tìm kiếm ngữ nghĩa và đồ thị tri thức):\n' + context if context else 'Không tìm thấy tài liệu liên quan. Hãy trả lời dựa trên kiến thức chung.'}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("human", system),
                MessagesPlaceholder("history"),
                ("human", "{query}"),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"history": history, "query": query})

    # ── Public API ─────────────────────────────────────────────────────────

    def respond(
        self, query: str, history: Optional[List[BaseMessage]] = None
    ) -> Dict[str, Any]:
        if not self._initialized:
            self.startup()
        history = history or []
        vector_docs = self._retrieve_vector(query)
        graph_docs = self._retrieve_graph(query)
        fused_docs = self._fuse_and_rerank(vector_docs, graph_docs)
        answer = self._generate(query, fused_docs, history)
        sources = [
            {
                "content": d.page_content[:300],
                "metadata": d.metadata,
                "source": "graph" if d in graph_docs else "vector",
            }
            for d in fused_docs[: self.top_k]
        ]
        return {"answer": answer, "sources": sources, "kg_docs_found": len(graph_docs)}

    async def graph_stream(
        self, query: str, history: Optional[List[BaseMessage]] = None
    ) -> AsyncGenerator[Dict, None]:
        """Stream GraphRAG response as SSE-compatible dicts."""
        if not self._initialized:
            self.startup()
        loop = asyncio.get_event_loop()

        yield {"type": "status", "data": "🕸️ Đang tra cứu đồ thị tri thức..."}

        result = await loop.run_in_executor(None, partial(self.respond, query, history))

        kg_count = result.get("kg_docs_found", 0)
        if kg_count > 0:
            yield {
                "type": "status",
                "data": f"📎 Tìm thấy {kg_count} tài liệu qua đồ thị tri thức",
            }

        yield {"type": "sources", "data": result["sources"]}

        answer = result["answer"]
        chunk_size = 8
        for i in range(0, len(answer), chunk_size):
            yield {"type": "chunk", "data": answer[i : i + chunk_size]}
            await asyncio.sleep(0)

        yield {"type": "done", "data": ""}


# Singleton
_graph_pipeline: Optional[GraphRAGPipeline] = None


def get_graph_pipeline() -> GraphRAGPipeline:
    global _graph_pipeline
    if _graph_pipeline is None:
        _graph_pipeline = GraphRAGPipeline(
            retrieval_k=config.RETRIEVAL_K,
            top_k=config.TOP_K,
        )
        _graph_pipeline.startup()
    return _graph_pipeline
