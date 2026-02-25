"""Embedding adapters: Vietnamese BGE-M3 dense + sparse for LangChain / Qdrant."""

from typing import List, Dict
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings
from qdrant_client.http import models as rest


def _to_token_id_weights(
    lexical_weights: Dict[str, float], tokenizer
) -> Dict[int, float]:
    unused = {
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    }
    result: Dict[int, float] = {}
    for token_id_str, weight in lexical_weights.items():
        tid = int(token_id_str)
        if tid not in unused and weight > 0:
            if tid not in result or weight > result[tid]:
                result[tid] = float(weight)
    return result


class VietnameseEmbedding:
    """Wrapper for AITeamVN/Vietnamese_Embedding_v2 using FlagEmbedding BGEM3."""

    def __init__(
        self, model_name: str = "AITeamVN/Vietnamese_Embedding_v2", device: str = "cpu"
    ):
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(
            model_name, use_fp16=(device == "cuda"), device=device
        )
        self.model.model.eval()
        self.tokenizer = self.model.tokenizer

    def encode(
        self,
        texts: List[str],
        return_dense=True,
        return_sparse=True,
        batch_size=8,
        max_length=512,
    ):
        if isinstance(texts, str):
            texts = [texts]
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


class VietnameseDenseAdapter(Embeddings):
    """LangChain-compatible dense embedding adapter."""

    def __init__(self, model: VietnameseEmbedding):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = self.model.encode(texts, return_dense=True, return_sparse=False)
        vecs = res["dense_embeddings"]
        return vecs.tolist() if hasattr(vecs, "tolist") else list(vecs)

    def embed_query(self, text: str) -> List[float]:
        res = self.model.encode([text], return_dense=True, return_sparse=False)
        vecs = res["dense_embeddings"]
        v = vecs[0]
        return v.tolist() if hasattr(v, "tolist") else list(v)


class VietnameseSparseAdapter(SparseEmbeddings):
    """LangChain-compatible sparse embedding adapter for Qdrant."""

    def __init__(self, model: VietnameseEmbedding):
        self.model = model

    def _to_sparse_vec(self, lw: Dict) -> rest.SparseVector:
        tid_weights = _to_token_id_weights(lw, self.model.tokenizer)
        idxs = sorted(tid_weights)
        return rest.SparseVector(indices=idxs, values=[tid_weights[i] for i in idxs])

    def embed_documents(self, texts: List[str]) -> List[rest.SparseVector]:
        res = self.model.encode(texts, return_dense=False, return_sparse=True)
        return [self._to_sparse_vec(lw) for lw in res["sparse_embeddings"]]

    def embed_query(self, text: str) -> rest.SparseVector:
        res = self.model.encode([text], return_dense=False, return_sparse=True)
        return self._to_sparse_vec(res["sparse_embeddings"][0])
