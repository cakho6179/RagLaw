"""Law document data loader — adapted for the legal_corpus.json array schema."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Vietnamese legal doc types for metadata tagging
_DOC_TYPE_PATTERN = re.compile(
    r"^(Luật|Bộ luật|Nghị định|Nghị quyết|Thông tư|Pháp lệnh|Quyết định|Hiến pháp|Thông tư liên tịch)",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"/(\d{4})/")


def _parse_law_id(law_id: str) -> Dict[str, str]:
    """Parse law_id like '01/2009/tt-bnn' or 'Luật 59/2020/QH14' into metadata."""
    doc_type = "Văn bản pháp luật"
    doc_number = law_id.strip()

    parts = law_id.split(maxsplit=1)
    if len(parts) > 1 and _DOC_TYPE_PATTERN.match(parts[0]):
        m = _DOC_TYPE_PATTERN.match(law_id)
        if m:
            doc_type = m.group(1).capitalize()
            doc_number = law_id[len(doc_type) :].strip()
    elif "/" in law_id:
        # e.g., "01/2009/tt-bnn" -> type "Thông tư"
        if "tt" in law_id.lower():
            doc_type = "Thông tư"
        elif "nđ" in law_id.lower() or "nd" in law_id.lower():
            doc_type = "Nghị định"
        elif "qđ" in law_id.lower() or "qd" in law_id.lower():
            doc_type = "Quyết định"

    year_m = _YEAR_PATTERN.search(doc_number)

    return {
        "document_id": law_id,
        "document_type": doc_type,
        "document_number": doc_number,
        "document_title": f"{doc_type} số {doc_number}".strip(),
        "year": year_m.group(1) if year_m else "",
    }


def chunk_corpus_law(law_dict: Dict) -> List[Document]:
    """Parse a single law dictionary from legal_corpus.json into LangChain Documents."""
    law_id = law_dict.get("law_id", "")
    articles = law_dict.get("articles", [])

    if not articles:
        return []

    meta = _parse_law_id(law_id)
    documents: List[Document] = []

    for item in articles:
        article_id = item.get("article_id", "")
        title = item.get("title", "")
        text = item.get("text", "")

        # Format content nicely
        content = f"{title}\n{text}".strip()
        if len(content) < 30:
            continue

        doc_meta = {
            **meta,
            "article_number": article_id,
            "article_title": title,
        }

        documents.append(Document(page_content=content, metadata=doc_meta))

    return documents


def iter_law_documents(data_dir: str = "data") -> Generator[List[Document], None, None]:
    """
    Yield chunked document lists for every law object in legal_corpus.json.
    Yields chunks for one law at a time to keep memory usage low during ingestion.
    """
    root = Path(data_dir)
    # Search for all json files, especially legal_corpus.json
    json_files = list(root.glob("**/*.json"))
    # Exclude non-corpus files
    json_files = [
        f
        for f in json_files
        if "law_graph.json" not in f.name and "eng_config" not in str(f)
    ]

    logger.info(f"Found {len(json_files)} law JSON files in {data_dir}")

    for fp in json_files:
        logger.info(f"Processing {fp.name}...")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                # Array of laws schema (legal_corpus.json)
                for law_dict in data:
                    chunks = chunk_corpus_law(law_dict)
                    if chunks:
                        yield chunks
            elif isinstance(data, dict):
                # Backwards compatible for single dict schema
                chunks = chunk_corpus_law(
                    {
                        "law_id": data.get("Id", "unknown"),
                        "articles": [
                            {
                                "article_id": "",
                                "title": "",
                                "text": data.get("Content", ""),
                            }
                        ],
                    }
                )
                if chunks:
                    yield chunks

        except Exception as e:
            logger.warning(f"Failed to process {fp.name}: {e}")


def load_all_documents(data_dir: str = "data") -> List[Document]:
    """Load and chunk all law documents into a flat list."""
    all_docs: List[Document] = []
    for chunks in iter_law_documents(data_dir):
        all_docs.extend(chunks)
    logger.info(f"Total chunks loaded: {len(all_docs)}")
    return all_docs
