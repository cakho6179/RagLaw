"""Qdrant vector store helpers."""
import logging
from qdrant_client import QdrantClient
from backend.config import config
logger = logging.getLogger(__name__)

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

def list_collections():
    client = get_qdrant_client()
    return [c.name for c in client.get_collections().collections]

def collection_exists(name: str) -> bool:
    return name in list_collections()
