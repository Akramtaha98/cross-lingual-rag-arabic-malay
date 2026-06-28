"""
retrieval/faiss_index.py
Build, save, and load FAISS indexes for dense retrieval.
"""

import os
import json
import logging
import numpy as np
import faiss
from typing import List, Dict, Tuple
from config import INDEX_DIR

# MPS + OpenMP interaction on Apple Silicon causes a SIGSEGV inside libomp when
# FAISS tries to spawn worker threads after a sentence-transformers MPS encode.
# Capping to 1 thread avoids the crash with no meaningful throughput loss on CPU.
faiss.omp_set_num_threads(1)

logger = logging.getLogger(__name__)


class FaissIndex:
    """
    Flat L2 FAISS index over passage embeddings.
    For larger corpora, swap to IndexIVFFlat (set nlist=100).
    """

    def __init__(self, dim: int, use_gpu: bool = False):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalizing
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            logger.info("FAISS running on GPU")
        self.passages: List[Dict] = []

    def add(self, embeddings: np.ndarray, passages: List[Dict]):
        """
        embeddings: (N, dim) float32, already L2-normalized
        passages:   list of passage dicts (same order as embeddings)
        """
        assert embeddings.shape[1] == self.dim, (
            f"Embedding dim mismatch: got {embeddings.shape[1]}, expected {self.dim}"
        )
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.passages.extend(passages)
        logger.info(f"Added {len(passages)} passages. Total: {len(self.passages)}")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        query_embedding: (1, dim) float32
        Returns list of (passage_dict, score) tuples, descending by score.
        """
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.passages[idx], float(score)))
        return results

    def save(self, name: str):
        """Save index and passages to INDEX_DIR/{name}.{faiss,json}."""
        os.makedirs(INDEX_DIR, exist_ok=True)
        index_path   = os.path.join(INDEX_DIR, f"{name}.faiss")
        passage_path = os.path.join(INDEX_DIR, f"{name}_passages.json")

        cpu_index = faiss.index_gpu_to_cpu(self.index) if hasattr(self.index, "getDevice") else self.index
        faiss.write_index(cpu_index, index_path)

        with open(passage_path, "w", encoding="utf-8") as f:
            json.dump(self.passages, f, ensure_ascii=False)

        logger.info(f"Saved FAISS index → {index_path}")

    @classmethod
    def load(cls, name: str, dim: int) -> "FaissIndex":
        index_path   = os.path.join(INDEX_DIR, f"{name}.faiss")
        passage_path = os.path.join(INDEX_DIR, f"{name}_passages.json")

        obj = cls(dim)
        obj.index = faiss.read_index(index_path)

        with open(passage_path, "r", encoding="utf-8") as f:
            obj.passages = json.load(f)

        logger.info(
            f"Loaded FAISS index from {index_path}: {len(obj.passages)} passages"
        )
        return obj
