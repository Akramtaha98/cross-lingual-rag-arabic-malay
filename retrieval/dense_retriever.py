"""
retrieval/dense_retriever.py
Dense retrieval using LaBSE, mE5, mE5-large, and BGE-M3 embeddings.

LaBSE    : Feng et al. (2022) — strong multilingual, great for Arabic/Malay
mE5-base : Wang et al. (2024) — multilingual E5 (278M params)
mE5-large: Wang et al. (2024) — larger mE5 (560M params), stronger baseline
BGE-M3   : Chen et al. (2024) — hybrid dense+sparse, current SOTA multilingual retrieval

All four are cross-lingual by design: a query in Arabic can retrieve Malay
passages and vice versa — which is exactly what EXP 2 and EXP 5 test.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from retrieval.faiss_index import FaissIndex
from config import RETRIEVER_MODELS, INDEX_DIR, DEFAULT_TOP_K

logger = logging.getLogger(__name__)

# mE5 and mE5-large require a task prefix; LaBSE and BGE-M3 do not.
ME5_QUERY_PREFIX   = "query: "
ME5_PASSAGE_PREFIX = "passage: "

# Models that require mE5-style prefixes
_ME5_FAMILY = {"me5", "me5large"}

# Safe batch sizes per model — large models OOM on MPS at batch_size=128
# BGE-M3 is ~570M params with colbert heads; mE5-large is ~560M params
_MODEL_BATCH_SIZES = {
    "labse":    128,
    "me5":      128,
    "me5large":  16,   # 1024-dim, large; reduce to avoid MPS OOM
    "bgem3":     8,    # largest model; colbert heads add memory overhead
}


class DenseRetriever:
    """
    Unified dense retriever for LaBSE, mE5, mE5-large, and BGE-M3.
    Encodes all corpus passages once and stores them in FAISS.
    """

    def __init__(self, model_name: str = "labse", batch_size: int = None, device: str = None):
        assert model_name in RETRIEVER_MODELS, (
            f"model_name must be one of {list(RETRIEVER_MODELS.keys())}"
        )
        self.model_name = model_name
        # Use model-specific safe batch size unless caller overrides
        self.batch_size = batch_size or _MODEL_BATCH_SIZES.get(model_name, 64)

        # BGE-M3 exceeds Apple M4 Pro MPS memory limit (30.19 GB) when loaded on GPU.
        # Force CPU for bgem3 and me5large to avoid RuntimeError: MPS backend out of memory.
        # CPU is slower (~20 min for 50k passages) but avoids the OOM crash entirely.
        if device is None:
            if model_name in {"bgem3", "me5large"}:
                device = "cpu"
                logger.info(
                    f"{model_name}: forcing device=cpu (model too large for MPS; "
                    "expect ~15–30 min for 50k passage index)"
                )
            else:
                device = None  # let sentence_transformers choose (MPS/CUDA/CPU)

        self.model = SentenceTransformer(RETRIEVER_MODELS[model_name], device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index: Optional[FaissIndex] = None
        logger.info(
            f"Loaded {model_name} (dim={self.dim}, batch_size={self.batch_size}, "
            f"device={self.model.device})"
        )

    def _prefix(self, texts: List[str], is_query: bool) -> List[str]:
        """Add mE5-style prefix for mE5 and mE5-large; BGE-M3 and LaBSE need none."""
        if self.model_name in _ME5_FAMILY:
            prefix = ME5_QUERY_PREFIX if is_query else ME5_PASSAGE_PREFIX
            return [prefix + t for t in texts]
        return texts

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        prefixed = self._prefix(texts, is_query)
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 200,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def index_corpus(
        self,
        passages: List[Dict],
        index_name: str,
        force_rebuild: bool = False,
    ) -> None:
        """
        Build (or load cached) FAISS index from passage list.
        passages: list of {"passage_id": ..., "text": ..., "title": ..., "lang": ...}
        """
        import os
        index_path = os.path.join(INDEX_DIR, f"{index_name}.faiss")

        if not force_rebuild and os.path.exists(index_path):
            logger.info(f"Loading cached index: {index_name}")
            self.index = FaissIndex.load(index_name, self.dim)
            return

        logger.info(f"Encoding {len(passages)} passages with {self.model_name} …")
        texts = [p["text"] for p in passages]

        # Encode in batches
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding passages"):
            batch = texts[i : i + self.batch_size]
            emb = self.encode(batch, is_query=False)
            all_embeddings.append(emb)
        embeddings = np.vstack(all_embeddings)

        self.index = FaissIndex(self.dim)
        self.index.add(embeddings, passages)
        self.index.save(index_name)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Tuple[Dict, float]]:
        """
        Retrieve top_k passages for a single query.
        Returns: [(passage_dict, score), ...] sorted descending by score
        """
        assert self.index is not None, "Call index_corpus() first"
        q_emb = self.encode([query], is_query=True)
        return self.index.search(q_emb, top_k)

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> List[List[Tuple[Dict, float]]]:
        """Retrieve for a list of queries."""
        assert self.index is not None, "Call index_corpus() first"
        q_embs = self.encode(queries, is_query=True)
        results = []
        for i in range(len(queries)):
            q_emb = q_embs[i : i + 1]
            results.append(self.index.search(q_emb, top_k))
        return results
