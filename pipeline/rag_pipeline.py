"""
pipeline/rag_pipeline.py
Full cross-lingual RAG pipeline.

Orchestrates retriever + generator for a given ExperimentConfig.
Supports:
  - Monolingual:   Arabic query → Arabic corpus
  - Cross-lingual: Arabic query → Malay corpus (and vice versa)
  - Mixed:         query → both corpora combined
"""

import logging
import time
from typing import List, Dict, Tuple, Optional
from config import ExperimentConfig, DEFAULT_TOP_K

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Cross-lingual Retrieval-Augmented Generation pipeline.
    """

    def __init__(
        self,
        retriever,          # BM25Retriever or DenseRetriever
        generator,          # MT5Generator
        config: ExperimentConfig,
    ):
        self.retriever = retriever
        self.generator = generator
        self.config = config

    def run_single(self, query: str) -> Dict:
        """
        Full RAG pipeline for one query.
        Returns dict with retrieved passages, generated answer, and timing.
        """
        t0 = time.time()

        # Step 1: Retrieve
        retrieved = self.retriever.retrieve(query, top_k=self.config.top_k)
        t_retrieve = time.time() - t0

        # Step 2: Generate
        answer = self.generator.generate(
            query=query,
            retrieved_passages=retrieved,
            top_k=self.config.top_k,
        )
        t_generate = time.time() - t0 - t_retrieve

        return {
            "query":      query,
            "answer":     answer,
            "passages":   retrieved,
            "t_retrieve": round(t_retrieve, 3),
            "t_generate": round(t_generate, 3),
        }

    def run_batch(
        self,
        queries: List[str],
        references: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> List[Dict]:
        """
        Run pipeline for a list of queries.
        Returns list of result dicts (same as run_single).
        """
        results = []
        for i in range(0, len(queries), batch_size):
            batch_q = queries[i : i + batch_size]
            batch_ref = references[i : i + batch_size] if references else [None] * len(batch_q)

            # Retrieve for all queries in batch
            batch_passages = [
                self.retriever.retrieve(q, top_k=self.config.top_k)
                for q in batch_q
            ]

            # Generate for all queries in batch
            batch_answers = self.generator.batch_generate(
                queries=batch_q,
                retrieved_passages_list=batch_passages,
                top_k=self.config.top_k,
                batch_size=batch_size,
            )

            for q, a, p, r in zip(batch_q, batch_answers, batch_passages, batch_ref):
                results.append({
                    "query":     q,
                    "answer":    a,
                    "reference": r,
                    "passages":  p,
                })

            logger.info(
                f"Processed {min(i + batch_size, len(queries))}/{len(queries)} queries"
            )

        return results


def build_pipeline(
    config: ExperimentConfig,
    corpora: Dict[str, List[Dict]],
    force_rebuild_index: bool = False,
) -> RAGPipeline:
    """
    Factory: build the right retriever + generator for a given ExperimentConfig.

    corpora: {"ar": [passage_dicts], "ms": [passage_dicts]}
    """
    from generation.mt5_generator import MT5Generator

    # ── Select corpus ──
    if config.corpus_lang == "both":
        passages = corpora["ar"] + corpora["ms"]
    else:
        passages = corpora[config.corpus_lang]

    # ── Build retriever ──
    if config.retriever == "bm25":
        from retrieval.bm25_retriever import BM25Retriever
        retriever = BM25Retriever(lang=config.query_lang)
        retriever.index_corpus(passages)

    else:  # "labse" or "me5"
        from retrieval.dense_retriever import DenseRetriever
        retriever = DenseRetriever(model_name=config.retriever)
        index_name = f"{config.retriever}_{config.corpus_lang}"
        retriever.index_corpus(
            passages,
            index_name=index_name,
            force_rebuild=force_rebuild_index,
        )

    # ── Build generator ──
    generator = MT5Generator(use_lora=config.use_lora)

    return RAGPipeline(retriever, generator, config)
