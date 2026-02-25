"""Qdrant ingestion script — embed law documents and upsert to Qdrant.

Usage:
  python -m backend.rag.ingest_qdrant                  # ingest all from data/
  python -m backend.rag.ingest_qdrant --data-dir path  # custom path
  python -m backend.rag.ingest_qdrant --dry-run        # count only
  python -m backend.rag.ingest_qdrant --recreate       # drop + recreate collection
"""

import argparse
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from backend.config import config
from backend.rag.data_loader import iter_law_documents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _ensure_collection(
    client: QdrantClient, collection: str, dim: int, recreate: bool
) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if recreate and collection in collections:
        client.delete_collection(collection)
        logger.info(f"Deleted existing collection '{collection}'")
        collections = []

    if collection not in collections:
        client.create_collection(
            collection_name=collection,
            vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
        )
        logger.info(f"Created collection '{collection}' (dim={dim})")
    else:
        logger.info(f"Using existing collection '{collection}'")


def ingest(
    data_dir: str = "data",
    collection: str | None = None,
    batch_size: int = 32,
    dry_run: bool = False,
    recreate: bool = False,
) -> None:
    coll = collection or config.COLLECTION_NAME
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    # Load embedding model
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    from backend.rag.adapters import VietnameseEmbedding, VietnameseDenseAdapter

    vn_emb = VietnameseEmbedding(
        model_name=config.EMBEDDING_MODEL, device=config.DEVICE
    )
    adapter = VietnameseDenseAdapter(vn_emb)

    if not dry_run:
        _ensure_collection(client, coll, config.EMBEDDING_DIM, recreate)

    total_docs = 0
    total_chunks = 0
    batch_texts: list[str] = []
    batch_payloads: list[dict] = []

    def flush_batch():
        nonlocal total_chunks
        if not batch_texts:
            return
        vectors = adapter.embed_documents(batch_texts)
        points = [
            rest.PointStruct(
                id=total_chunks - len(batch_texts) + i + 1,
                vector=vec,
                payload=payload,
            )
            for i, (vec, payload) in enumerate(zip(vectors, batch_payloads))
        ]
        client.upsert(collection_name=coll, points=points)
        logger.info(f"  Upserted batch of {len(points)} points (total={total_chunks})")
        batch_texts.clear()
        batch_payloads.clear()

    for doc_chunks in iter_law_documents(data_dir):
        total_docs += 1
        for chunk in doc_chunks:
            total_chunks += 1
            if dry_run:
                continue
            batch_texts.append(chunk.page_content)
            batch_payloads.append({**chunk.metadata, "text": chunk.page_content})
            if len(batch_texts) >= batch_size:
                flush_batch()

    if not dry_run:
        flush_batch()  # Flush remaining

    logger.info(f"\n{'[DRY RUN] ' if dry_run else ''}✅ Ingestion complete")
    logger.info(f"  Documents: {total_docs}  |  Chunks: {total_chunks}")
    if not dry_run:
        info = client.get_collection(coll)
        logger.info(f"  Qdrant collection '{coll}': {info.points_count} points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate collection before ingestion",
    )
    args = parser.parse_args()
    ingest(
        data_dir=args.data_dir,
        collection=args.collection,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        recreate=args.recreate,
    )
