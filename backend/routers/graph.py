"""Graph router — Knowledge Graph stats, entity search, rebuild, and GraphRAG stream."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session as DbSession

from backend.core.database import get_db
from backend.services.session_service import session_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/graph", tags=["graph"])


class GraphChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


# ── KG Info ────────────────────────────────────────────────────────────────


@router.get("/stats")
async def graph_stats():
    """Return Knowledge Graph node/edge statistics."""
    try:
        from backend.rag.knowledge_graph import get_knowledge_graph

        kg = get_knowledge_graph()
        return kg.get_stats()
    except Exception as e:
        return {"error": str(e), "nodes": 0, "edges": 0}


@router.get("/entity/{name}")
async def get_entity_neighbours(name: str, hops: int = Query(2, ge=1, le=3)):
    """Get all nodes within N hops of a named entity."""
    from backend.rag.knowledge_graph import get_knowledge_graph

    kg = get_knowledge_graph()
    node_id = f"concept:{name.lower().strip()}"
    neighbours = kg.get_neighbours(node_id, hops=hops)
    return {"entity": name, "neighbours": neighbours, "total": len(neighbours)}


@router.post("/build")
async def rebuild_graph(
    background_tasks: BackgroundTasks, use_llm: bool = False, data_dir: str = "data"
):
    """Trigger a full Knowledge Graph rebuild in the background."""

    def do_rebuild():
        from backend.rag.knowledge_graph import rebuild_knowledge_graph

        kg = rebuild_knowledge_graph(data_dir=data_dir, use_llm=use_llm)
        logger.info(f"KG rebuild complete: {kg.get_stats()}")

    background_tasks.add_task(do_rebuild)
    return {
        "status": "rebuilding",
        "message": "Knowledge Graph rebuild started in background",
    }


# ── GraphRAG Stream ────────────────────────────────────────────────────────


@router.post("/stream")
async def graph_rag_stream(
    payload: GraphChatRequest,
    db: DbSession = Depends(get_db),
):
    """GraphRAG streaming endpoint — vector + KG hybrid retrieval via SSE."""
    session_id = payload.session_id
    is_new = False
    if not session_id:
        sess = session_service.create_session(db)
        session_id = sess.id
        is_new = True
    else:
        sess = session_service.get_session(db, session_id)
        if not sess:
            sess = session_service.create_session(db, title="GraphRAG Chat")
            session_id = sess.id
            is_new = True

    try:
        from backend.rag.graph_pipeline import get_graph_pipeline

        pipeline = get_graph_pipeline()
    except Exception as e:
        logger.error(f"GraphRAG init error: {e}")

        async def err_gen():
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

        return StreamingResponse(err_gen(), media_type="text/event-stream")

    async def event_generator():
        full_answer = ""
        all_sources = []
        async for event in pipeline.graph_stream(payload.message):
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


# ── Multi-Tool Agent Stream ────────────────────────────────────────────────


@router.post("/agent/stream")
async def agent_stream(
    payload: GraphChatRequest,
    db: DbSession = Depends(get_db),
):
    """Multi-tool LangGraph agent stream — 4 tools: vector, KG, lookup, text search."""
    session_id = payload.session_id
    is_new = False
    if not session_id:
        sess = session_service.create_session(db)
        session_id = sess.id
        is_new = True

    try:
        from backend.agents.query_analyzer import get_agent

        agent = get_agent()
    except Exception as e:
        logger.error(f"Agent init error: {e}")

        async def err_gen():
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

        return StreamingResponse(err_gen(), media_type="text/event-stream")

    async def event_generator():
        full_answer = ""
        all_sources = []
        async for event in agent.arespond_stream(payload.message):
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
