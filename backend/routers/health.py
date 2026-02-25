"""Health check router."""

from fastapi import APIRouter
from backend.config import config

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "app": config.APP_NAME,
        "model": config.GEMINI_MODEL,
        "collection": config.COLLECTION_NAME,
    }


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}
