"""Law document lookup service — adapted for legal_corpus.json list schema."""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import config

logger = logging.getLogger(__name__)

DOCS_ROOT_DEFAULT = Path("data")


@dataclass
class LawDocument:
    id: str
    title: str
    type: str
    number: str
    date: str
    snippet: str
    content: str
    category: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LawArticle:
    article_number: str
    article_title: str
    content: str
    chapter: Optional[str] = None
    section: Optional[str] = None


def _parse_law_id(law_id: str) -> Dict[str, str]:
    doc_type = "Văn bản pháp luật"
    doc_number = law_id.strip()

    parts = law_id.split(maxsplit=1)
    if len(parts) > 1 and re.match(
        r"^(Luật|Bộ luật|Nghị định|Nghị quyết|Thông tư|Pháp lệnh|Quyết định)",
        parts[0],
        re.IGNORECASE,
    ):
        m = re.match(
            r"^(Luật|Bộ luật|Nghị định|Nghị quyết|Thông tư|Pháp lệnh|Quyết định)",
            law_id,
            re.IGNORECASE,
        )
        if m:
            doc_type = m.group(1).capitalize()
            doc_number = law_id[len(doc_type) :].strip()
    elif "/" in law_id:
        if "tt" in law_id.lower():
            doc_type = "Thông tư"
        elif "nđ" in law_id.lower() or "nd" in law_id.lower():
            doc_type = "Nghị định"
        elif "qđ" in law_id.lower() or "qd" in law_id.lower():
            doc_type = "Quyết định"

    year_m = re.search(r"/(\d{4})/", doc_number)

    return {
        "id": law_id,
        "type": doc_type,
        "number": doc_number,
        "year": year_m.group(1) if year_m else "",
        "category": "law",
        "title": f"{doc_type} số {doc_number}".strip() if doc_number else doc_type,
    }


class LawDocumentService:
    """Loads and indexes law documents from legal_corpus.json."""

    def __init__(self, docs_root: Optional[str] = None):
        self.docs_root = Path(docs_root) if docs_root else DOCS_ROOT_DEFAULT
        self._documents: Dict[
            str, Dict
        ] = {}  # In-memory storage of all laws to avoid huge mem usage
        self._articles: Dict[str, List[Dict]] = {}
        self._documents_cache: List[Dict] = []
        self._types_cache: List[str] = []

    def startup(self) -> None:
        logger.info(f"LawDocumentService: loading from {self.docs_root}")
        self._build_documents_cache()
        logger.info(f"  → {len(self._documents)} laws indexed in memory")

    def _build_documents_cache(self) -> None:
        if self._documents_cache:
            return

        types_set = set()

        # Find json files
        json_files = [
            f
            for f in self.docs_root.glob("**/*.json")
            if "law_graph" not in f.name and "eng_config" not in str(f)
        ]

        for fp in json_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    continue

                for law_dict in data:
                    law_id = law_dict.get("law_id")
                    if not law_id:
                        continue

                    articles = law_dict.get("articles", [])
                    meta = _parse_law_id(law_id)

                    # Compute snippet and full text
                    snippet = ""
                    content_full = []
                    parsed_articles = []

                    for a in articles:
                        art_id = a.get("article_id", "")
                        title = a.get("title", "")
                        text = a.get("text", "")

                        full_art = f"{title}\n{text}"
                        content_full.append(full_art)
                        parsed_articles.append(
                            {
                                "article_number": art_id,
                                "article_title": title,
                                "content": full_art,
                                "chapter": None,
                                "section": None,
                            }
                        )

                        if not snippet and len(text) > 30:
                            snippet = text[:200] + "..." if len(text) > 200 else text

                    content_str = "\n\n".join(content_full)

                    doc = LawDocument(
                        id=law_id,
                        title=meta["title"],
                        type=meta["type"],
                        number=meta["number"],
                        date=meta["year"],
                        snippet=snippet,
                        content=content_str,
                        category="laws",
                    )

                    self._documents[law_id] = format_for_frontend(doc)
                    self._articles[law_id] = parsed_articles

                    self._documents_cache.append(
                        {
                            "id": doc.id,
                            "title": doc.title,
                            "type": doc.type,
                            "ref": f"{doc.type} {doc.number}",
                            "date": doc.date,
                            "snippet": doc.snippet,
                            "content_lower": doc.content.lower(),
                        }
                    )
                    types_set.add(doc.type)

            except Exception as e:
                logger.warning(f"Error indexing {fp}: {e}")

        self._documents_cache.sort(key=lambda x: x.get("date", ""), reverse=True)
        self._types_cache = sorted(types_set)

    def get_document(self, doc_id: str) -> Optional[Dict]:
        return self._documents.get(doc_id)

    def get_articles(self, doc_id: str) -> List[Dict]:
        return self._articles.get(doc_id, [])

    def list_documents(
        self, doc_type: Optional[str] = None, page: int = 1, page_size: int = 20
    ) -> Dict:
        docs = [
            d
            for d in self._documents_cache
            if not doc_type or d.get("type") == doc_type
        ]
        total = len(docs)
        items = docs[(page - 1) * page_size : page * page_size]
        items = [
            {k: v for k, v in item.items() if k != "content_lower"} for item in items
        ]
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 0,
        }

    def search_documents(
        self,
        query: str,
        doc_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict:
        q = query.lower()
        results = [
            d
            for d in self._documents_cache
            if (not doc_type or d.get("type") == doc_type)
            and (q in d.get("title", "").lower() or q in d.get("content_lower", ""))
        ]
        total = len(results)
        items = results[(page - 1) * page_size : page * page_size]
        items = [
            {k: v for k, v in item.items() if k != "content_lower"} for item in items
        ]
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 0,
        }

    def get_types(self) -> List[str]:
        return self._types_cache

    def search_by_content(self, source_text: str) -> Optional[Dict]:
        if not source_text or len(source_text) < 50:
            return None
        m = re.search(r"\d+\.\s+(.{50,120})", source_text)
        needle = m.group(1).strip() if m else None

        sentences = []
        if not needle:
            sentences = re.findall(
                r"[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][^.!?]{50,150}",
                source_text,
            )
            if not sentences:
                return None

        for doc in self._documents_cache:
            content_lower = doc.get("content_lower", "")
            if needle and needle.lower() in content_lower:
                return self.get_document(doc["id"])
            for sentence in sentences:
                if sentence.lower() in content_lower:
                    return self.get_document(doc["id"])
        return None


def format_for_frontend(doc: LawDocument) -> Dict:
    return {
        "id": doc.id,
        "title": doc.title,
        "type": doc.type,
        "ref": f"{doc.type} {doc.number}",
        "date": doc.date,
        "snippet": doc.snippet,
        "content": doc.content,
    }


# Singleton
_lookup_service: Optional[LawDocumentService] = None


def get_lookup_service() -> LawDocumentService:
    global _lookup_service
    if _lookup_service is None:
        _lookup_service = LawDocumentService()
        _lookup_service.startup()
    return _lookup_service
