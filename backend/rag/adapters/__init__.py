"""Vietnamese Embedding Adapters for LangChain + Qdrant.

Auto-detects model type:
  - BGE-M3 family (AITeamVN/Vietnamese_Embedding_v2, BAAI/bge-m3): uses FlagEmbedding BGEM3FlagModel
  - All other models: falls back to SentenceTransformer (dense only)

Exports:
  VietnameseEmbedding       — unified wrapper, auto-selects backend
  VietnameseDenseAdapter    — LangChain Embeddings interface (dense vectors)
  VietnameseSparseAdapter   — LangChain SparseEmbeddings interface (sparse, BGE-M3 only)
"""

import logging
from typing import Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings
from qdrant_client.http import models as rest

logger = logging.getLogger(__name__)

# Model names known to use the BGEM3FlagModel backend
_BGE_M3_PREFIXES = (
    "BAAI/bge-m3",
    "AITeamVN/Vietnamese_Embedding_v2",
    "bge-m3",
)


def _is_bgem3_model(model_name: str) -> bool:
    return any(model_name.startswith(p) or p in model_name for p in _BGE_M3_PREFIXES)


def _to_token_id_weights(lexical_weights: Dict, tokenizer) -> Dict[int, float]:
    unused = {
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    }
    result: Dict[int, float] = {}
    for tid_str, weight in lexical_weights.items():
        tid = int(tid_str)
        if tid not in unused and weight > 0:
            if tid not in result or weight > result[tid]:
                result[tid] = float(weight)
    return result


class VietnameseEmbedding:
    """Unified Vietnamese embedding wrapper.

    For BGE-M3 models: uses FlagEmbedding.BGEM3FlagModel (dense + sparse)
    For other models:  uses SentenceTransformer (dense only)
    """

    def __init__(
        self, model_name: str = "AITeamVN/Vietnamese_Embedding_v2", device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._backend: str = "bgem3" if _is_bgem3_model(model_name) else "sbert"
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        if self._backend == "bgem3":
            try:
                from FlagEmbedding import BGEM3FlagModel

                self.model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=(self.device in ("cuda", "gpu")),
                    device=self.device if self.device not in ("gpu",) else "cuda",
                )
                self.model.model.eval()
                self.tokenizer = self.model.tokenizer
                logger.info(f"Loaded BGE-M3 model: {self.model_name}")
                return
            except ImportError:
                logger.warning(
                    "FlagEmbedding not available, falling back to SentenceTransformer"
                )
                self._backend = "sbert"
            except Exception as e:
                logger.warning(
                    f"BGE-M3 load failed ({e}), falling back to SentenceTransformer"
                )
                self._backend = "sbert"

        # SentenceTransformer fallback
        from sentence_transformers import SentenceTransformer

        torch_device = "cuda" if self.device in ("cuda", "gpu") else "cpu"
        self.model = SentenceTransformer(self.model_name, device=torch_device)
        logger.info(
            f"Loaded SentenceTransformer model: {self.model_name} on {torch_device}"
        )

    def encode(
        self,
        texts: List[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> Dict:
        if isinstance(texts, str):
            texts = [texts]

        if self._backend == "bgem3":
            import torch

            with torch.no_grad():
                res = self.model.encode(
                    texts,
                    batch_size=min(batch_size, len(texts)),
                    max_length=max_length,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert_vecs=False,
                )
            out = {}
            if return_dense:
                out["dense_embeddings"] = res["dense_vecs"]
            if return_sparse:
                out["sparse_embeddings"] = res["lexical_weights"]
            return out
        else:
            # SentenceTransformer: dense only
            vecs = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return {"dense_embeddings": vecs}

    @property
    def supports_sparse(self) -> bool:
        return self._backend == "bgem3"

    @property
    def embedding_dim(self) -> int:
        if self._backend == "bgem3":
            return self.model.model.config.hidden_size
        else:
            return self.model.get_sentence_embedding_dimension()


class VietnameseDenseAdapter(Embeddings):
    """LangChain-compatible dense embedding adapter — works with any backend."""

    def __init__(self, model: VietnameseEmbedding):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = self.model.encode(texts, return_dense=True, return_sparse=False)
        vecs = res["dense_embeddings"]
        return vecs.tolist() if hasattr(vecs, "tolist") else [list(v) for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        res = self.model.encode([text], return_dense=True, return_sparse=False)
        v = res["dense_embeddings"][0]
        return v.tolist() if hasattr(v, "tolist") else list(v)


class VietnameseSparseAdapter(SparseEmbeddings):
    """LangChain-compatible sparse embedding adapter — BGE-M3 only.

    Falls back to empty sparse vectors for non-BGE-M3 models.
    """

    def __init__(self, model: VietnameseEmbedding):
        self.model = model

    def _to_sparse_vec(self, lw: Dict) -> rest.SparseVector:
        if not self.model.tokenizer:
            return rest.SparseVector(indices=[], values=[])
        tid_weights = _to_token_id_weights(lw, self.model.tokenizer)
        idxs = sorted(tid_weights)
        return rest.SparseVector(indices=idxs, values=[tid_weights[i] for i in idxs])

    def embed_documents(self, texts: List[str]) -> List[rest.SparseVector]:
        if not self.model.supports_sparse:
            return [rest.SparseVector(indices=[], values=[]) for _ in texts]
        res = self.model.encode(texts, return_dense=False, return_sparse=True)
        return [self._to_sparse_vec(lw) for lw in res["sparse_embeddings"]]

    def embed_query(self, text: str) -> rest.SparseVector:
        if not self.model.supports_sparse:
            return rest.SparseVector(indices=[], values=[])
        res = self.model.encode([text], return_dense=False, return_sparse=True)
        return self._to_sparse_vec(res["sparse_embeddings"][0])


__all__ = ["VietnameseEmbedding", "VietnameseDenseAdapter", "VietnameseSparseAdapter"]
