"""Documents router — full CRUD using LawDocumentService."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from backend.services.lookup_service import LawDocumentService, get_lookup_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
async def list_documents(
    doc_type: Optional[str] = Query(
        None, description="Filter by document type (Luật, Nghị định, ...)"
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: LawDocumentService = Depends(get_lookup_service),
):
    """List all law documents with optional type filter and pagination."""
    return service.list_documents(doc_type=doc_type, page=page, page_size=page_size)


@router.get("/types")
async def get_document_types(service: LawDocumentService = Depends(get_lookup_service)):
    """Get all distinct document types available in the database."""
    return {"types": service.get_types()}


@router.get("/search")
async def search_documents(
    q: str = Query(..., description="Search query"),
    doc_type: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: LawDocumentService = Depends(get_lookup_service),
):
    """Text-based search across document title and content."""
    return service.search_documents(
        query=q, doc_type=doc_type, page=page, page_size=page_size
    )


@router.get("/{doc_id}")
async def get_document(
    doc_id: str, service: LawDocumentService = Depends(get_lookup_service)
):
    """Get full content of a single law document by ID."""
    doc = service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return doc


@router.get("/{doc_id}/articles")
async def get_document_articles(
    doc_id: str, service: LawDocumentService = Depends(get_lookup_service)
):
    """Get all articles parsed from a law document."""
    articles = service.get_articles(doc_id)
    if articles is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return {"doc_id": doc_id, "articles": articles, "total": len(articles)}
