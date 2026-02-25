"""Knowledge Graph for Vietnamese Legal Documents.

Uses NetworkX DiGraph + LLMGraphTransformer to extract:
  Nodes: LegalDocument, Article, Concept, Organization
  Edges: CITES, AMENDS, IMPLEMENTS, DEFINES, REFERENCES

Also uses regex to extract cross-references between laws (Luật/NĐ numbers).
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

# ── Patterns for regex-based reference extraction ─────────────────────────

_LAW_REF_PATTERNS = [
    re.compile(r"Luật\s+(?:[\w\s]+?)\s+số\s+([\d]+/\d{4}/QH[\d]+)", re.IGNORECASE),
    re.compile(r"Nghị\s+định\s+số\s+([\d]+/\d{4}/NĐ-CP)", re.IGNORECASE),
    re.compile(r"Thông\s+tư\s+số\s+([\d]+/\d{4}/TT-\w+)", re.IGNORECASE),
    re.compile(r"Quyết\s+định\s+số\s+([\d]+/\d{4}/QĐ-\w+)", re.IGNORECASE),
    re.compile(
        r"Điều\s+(\d+)\s+của\s+(Luật|Nghị\s+định|Bộ\s+luật)\s+([\w\s]+)", re.IGNORECASE
    ),
]

_CONCEPT_KEYWORDS = [
    "quyền sở hữu",
    "hợp đồng",
    "trách nhiệm",
    "nghĩa vụ",
    "vi phạm",
    "bồi thường",
    "tội phạm",
    "hình phạt",
    "doanh nghiệp",
    "tài sản",
    "thừa kế",
    "ly hôn",
    "lao động",
    "thuế",
    "đất đai",
    "môi trường",
]


class KnowledgeGraph:
    """NetworkX-based knowledge graph for law documents."""

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._doc_id_to_node: Dict[str, str] = {}

    def add_document_node(self, doc_id: str, title: str, doc_type: str) -> str:
        node_id = f"doc:{doc_id}"
        self.graph.add_node(
            node_id, type="LegalDocument", doc_id=doc_id, title=title, doc_type=doc_type
        )
        self._doc_id_to_node[doc_id] = node_id
        return node_id

    def add_concept_node(self, concept: str) -> str:
        node_id = f"concept:{concept.lower().strip()}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, type="Concept", name=concept)
        return node_id

    def add_reference_edge(
        self, source_node: str, target_node: str, relation: str = "REFERENCES"
    ) -> None:
        if source_node != target_node:
            self.graph.add_edge(source_node, target_node, relation=relation)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": dict(
                zip(
                    *map(
                        list,
                        zip(
                            *[
                                (data.get("type", "unknown"), 1)
                                for _, data in self.graph.nodes(data=True)
                            ]
                        ),
                    )
                )
            )
            if self.graph.nodes
            else {},
        }

    def get_neighbours(self, node_id: str, hops: int = 2) -> List[Dict]:
        if not self.graph.has_node(node_id):
            return []
        visited: Set[str] = set()
        frontier = {node_id}
        result = []
        for _ in range(hops):
            next_frontier: Set[str] = set()
            for n in frontier:
                for nb in list(self.graph.successors(n)) + list(
                    self.graph.predecessors(n)
                ):
                    if nb not in visited:
                        next_frontier.add(nb)
                        data = dict(self.graph.nodes[nb])
                        data["node_id"] = nb
                        result.append(data)
                        visited.add(nb)
            frontier = next_frontier
        return result

    def get_related_documents(self, query_text: str, hops: int = 2) -> List[str]:
        matched_nodes: Set[str] = set()
        q_lower = query_text.lower()
        for _, data in self.graph.nodes(data=True):
            if data.get("type") == "Concept":
                name = data.get("name", "").lower()
                if name in q_lower or q_lower in name:
                    matched_nodes.add(f"concept:{name}")
        doc_ids: Set[str] = set()
        for node in matched_nodes:
            neighbours = self.get_neighbours(node, hops=hops)
            for nb in neighbours:
                if nb.get("type") == "LegalDocument":
                    doc_ids.add(nb.get("doc_id", ""))
        return list(doc_ids)

    def save(self, path: str = "data/law_graph.json") -> None:
        data = nx.node_link_data(self.graph)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Saved KG to {path} ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)"
        )

    def load(self, path: str = "data/law_graph.json") -> bool:
        p = Path(path)
        if not p.exists():
            return False
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        self._doc_id_to_node = {
            d.get("doc_id"): nid
            for nid, d in self.graph.nodes(data=True)
            if d.get("type") == "LegalDocument" and d.get("doc_id")
        }
        logger.info(
            f"Loaded KG from {path}: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        )
        return True


class KnowledgeGraphBuilder:
    def __init__(self, use_llm: bool = False):
        self.kg = KnowledgeGraph()
        self.use_llm = use_llm
        self._llm_transformer = None

    def _init_llm_transformer(self):
        try:
            from langchain_experimental.graph_transformers import LLMGraphTransformer
            from langchain_google_genai import ChatGoogleGenerativeAI
            from backend.config import config

            llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0,
            )
            self._llm_transformer = LLMGraphTransformer(
                llm=llm,
                allowed_node_types=[
                    "LegalDocument",
                    "Article",
                    "Concept",
                    "Organization",
                ],
                allowed_relationship_types=[
                    "CITES",
                    "AMENDS",
                    "IMPLEMENTS",
                    "DEFINES",
                    "REFERENCES",
                ],
            )
            logger.info("LLMGraphTransformer initialized")
        except Exception as e:
            logger.warning(f"LLMGraphTransformer unavailable: {e}")
            self.use_llm = False

    def _extract_references_regex(
        self, content: str, source_doc_id: str
    ) -> List[Tuple[str, str]]:
        refs = []
        for pattern in _LAW_REF_PATTERNS:
            for m in pattern.finditer(content):
                ref = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                if ref:
                    refs.append((source_doc_id, ref))
        return refs

    def _extract_concepts_keyword(self, content: str) -> List[str]:
        found = []
        content_lower = content.lower()
        for kw in _CONCEPT_KEYWORDS:
            if kw in content_lower:
                found.append(kw)
        return found

    def build_from_data_dir(
        self, data_dir: str = "data", limit: Optional[int] = None
    ) -> None:
        root = Path(data_dir)
        json_files = [
            f
            for f in root.glob("**/*.json")
            if "law_graph" not in f.name and "eng_config" not in str(f)
        ]
        logger.info(f"Building KG from {len(json_files)} files in {data_dir}")

        if self.use_llm:
            self._init_llm_transformer()

        doc_nodes: Dict[str, Tuple[str, str]] = {}
        all_refs: List[Tuple[str, str]] = []
        all_concepts: Dict[str, List[str]] = {}

        laws_processed = 0

        for fp in json_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for law_dict in data:
                        if limit and laws_processed >= limit:
                            break
                        law_id = law_dict.get("law_id")
                        if not law_id:
                            continue

                        articles = law_dict.get("articles", [])
                        content = "\n\n".join(a.get("text", "") for a in articles)

                        doc_type = "Văn bản"
                        if "/" in law_id:
                            if "tt" in law_id.lower():
                                doc_type = "Thông tư"
                            elif "nđ" in law_id.lower() or "nd" in law_id.lower():
                                doc_type = "Nghị định"
                            elif "qđ" in law_id.lower() or "qd" in law_id.lower():
                                doc_type = "Quyết định"
                        elif law_id.lower().startswith("luật"):
                            doc_type = "Luật"

                        doc_nodes[law_id] = (f"{doc_type} {law_id}", doc_type)
                        all_refs.extend(self._extract_references_regex(content, law_id))
                        all_concepts[law_id] = self._extract_concepts_keyword(content)
                        laws_processed += 1
                elif isinstance(data, dict):
                    # fallback
                    doc_id = str(data.get("Id", fp.stem))
                    content = data.get("Content", "")
                    doc_nodes[doc_id] = (f"Văn bản {doc_id}", "Văn bản")
                    all_refs.extend(self._extract_references_regex(content, doc_id))
                    all_concepts[doc_id] = self._extract_concepts_keyword(content)
                    laws_processed += 1
            except Exception as e:
                logger.debug(f"Skipped {fp}: {e}")
                continue

            if limit and laws_processed >= limit:
                break

        for doc_id, (title, doc_type) in doc_nodes.items():
            self.kg.add_document_node(doc_id, title, doc_type)

        for doc_id, concepts in all_concepts.items():
            src_node = self.kg._doc_id_to_node.get(doc_id)
            if not src_node:
                continue
            for concept in concepts:
                concept_node = self.kg.add_concept_node(concept)
                self.kg.add_reference_edge(src_node, concept_node, "DEFINES")

        for source_id, ref_num in all_refs:
            target_id = next(
                (did for did in doc_nodes if ref_num in did or did in ref_num), None
            )
            if target_id and target_id != source_id:
                src = self.kg._doc_id_to_node.get(source_id)
                tgt = self.kg._doc_id_to_node.get(target_id)
                if src and tgt:
                    self.kg.add_reference_edge(src, tgt, "CITES")

        stats = self.kg.get_stats()
        logger.info(f"✅ KG built: {stats['nodes']} nodes, {stats['edges']} edges")


_kg: Optional[KnowledgeGraph] = None


def get_knowledge_graph(
    data_dir: str = "data", graph_path: str = "data/law_graph.json"
) -> KnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
        if not _kg.load(graph_path):
            logger.info("No saved KG found. Building from data_dir...")
            builder = KnowledgeGraphBuilder(use_llm=False)
            builder.build_from_data_dir(data_dir)
            _kg = builder.kg
            try:
                _kg.save(graph_path)
            except Exception as e:
                logger.warning(f"Could not save KG: {e}")
    return _kg


def rebuild_knowledge_graph(
    data_dir: str = "data",
    graph_path: str = "data/law_graph.json",
    use_llm: bool = False,
    limit: Optional[int] = None,
) -> KnowledgeGraph:
    global _kg
    builder = KnowledgeGraphBuilder(use_llm=use_llm)
    builder.build_from_data_dir(data_dir, limit=limit)
    _kg = builder.kg
    _kg.save(graph_path)
    return _kg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()
    if args.build:
        kg = rebuild_knowledge_graph(
            args.data_dir, use_llm=args.use_llm, limit=args.limit
        )
        print(kg.get_stats())
