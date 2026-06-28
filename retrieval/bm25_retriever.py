"""
retrieval/bm25_retriever.py
BM25 sparse retrieval baseline (Robertson & Zaragoza, 2009).

Used in EXP 1 as the sparse baseline against dense retrievers.
Language-specific tokenization: whitespace for Arabic, word-level for Malay.
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)


def tokenize(text: str, lang: str = "ar") -> List[str]:
    """
    Simple whitespace tokenizer.
    For Arabic: whitespace is sufficient for BM25.
    For Malay: also whitespace (Latin script, similar to English).
    """
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r"[^\w\s؀-ۿ]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


class BM25Retriever:
    """
    BM25 baseline retriever.
    Must index all passages before calling retrieve().
    """

    def __init__(self, lang: str = "ar"):
        self.lang = lang
        self.passages: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None

    def index_corpus(self, passages: List[Dict]) -> None:
        """
        Build BM25 index from passage list.
        passages: [{"passage_id":..., "text":..., "lang":...}, ...]
        """
        logger.info(f"Building BM25 index for {len(passages)} passages …")
        self.passages = passages
        tokenized = [tokenize(p["text"], self.lang) for p in passages]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built.")

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Tuple[Dict, float]]:
        """
        Returns [(passage_dict, bm25_score), ...] sorted descending.
        """
        assert self.bm25 is not None, "Call index_corpus() first"
        tokens = tokenize(query, self.lang)
        scores = self.bm25.get_scores(tokens)

        # Get top_k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.passages[i], float(scores[i])) for i in top_indices]

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> List[List[Tuple[Dict, float]]]:
        return [self.retrieve(q, top_k) for q in queries]
