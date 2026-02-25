"""FastAPI Application Entry Point — RagLaw v5 (LangChain + ADK + GraphRAG)."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import config
from backend.core.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("backend.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info("🚀 Starting RagLaw API v5 (LangChain + ADK + GraphRAG)...")
    init_db()
    logger.info("✅ Database initialized (raglaw.db)")
    # Pipelines are initialized lazily on first request to keep startup fast
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("👋 RagLaw API shutting down")


app = FastAPI(
    title=config.APP_NAME,
    version="5.0.0",
    description=(
        "Vietnamese Legal RAG System — LangChain + Google ADK + Knowledge Graph\n\n"
        "**Endpoints:**\n"
        "- `/api/chat` — sync LangChain chat\n"
        "- `/api/chat/stream` — LangGraph agentic SSE stream\n"
        "- `/api/chat/adk/stream` — Google ADK agent SSE stream\n"
        "- `/api/graph/stream` — GraphRAG hybrid SSE stream 🆕\n"
        "- `/api/graph/agent/stream` — Multi-tool agent SSE stream 🆕\n"
        "- `/api/documents` — Law document lookup 🆕\n"
        "- `/api/graph/stats` — Knowledge Graph statistics 🆕\n"
        "- `/api/sessions` — Anonymous chat session CRUD"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Routers ────────────────────────────────────────────────────────────────

from backend.routers import chat, sessions, health, documents, graph  # noqa: E402

app.include_router(chat.router, prefix=config.API_STR)
app.include_router(sessions.router, prefix=config.API_STR)
app.include_router(health.router, prefix=config.API_STR)
app.include_router(documents.router, prefix=config.API_STR)
app.include_router(graph.router, prefix=config.API_STR)


@app.get("/")
async def root():
    return {
        "status": "online",
        "version": "5.0.0",
        "docs": "/docs",
        "features": [
            "LangGraph Agentic RAG",
            "Google ADK Agent",
            "Knowledge Graph (NetworkX)",
            "GraphRAG Hybrid Retrieval",
            "Multi-tool Agent (4 tools)",
            "Law Document Lookup",
        ],
        "endpoints": {
            "chat_sync": f"{config.API_STR}/chat",
            "chat_stream": f"{config.API_STR}/chat/stream",
            "adk_stream": f"{config.API_STR}/chat/adk/stream",
            "graph_stream": f"{config.API_STR}/graph/stream",
            "agent_stream": f"{config.API_STR}/graph/agent/stream",
            "documents": f"{config.API_STR}/documents",
            "graph_stats": f"{config.API_STR}/graph/stats",
            "sessions": f"{config.API_STR}/sessions",
            "health": f"{config.API_STR}/health",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8001, reload=True)
