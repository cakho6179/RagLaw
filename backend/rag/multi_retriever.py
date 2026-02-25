"""Multi-retriever combining dense + sparse results."""
import logging
from typing import List
from langchain_core.documents import Document
logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(results_list: List[List[Document]], k: int = 60) -> List[Document]:
    scores = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    seen = set()
    merged = []
    for results in results_list:
        for doc in results:
            key = doc.page_content[:100]
            if key in sorted_keys and key not in seen:
                merged.append(doc)
                seen.add(key)
    return merged
