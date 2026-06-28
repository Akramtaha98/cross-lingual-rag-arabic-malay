"""
evaluation/metrics.py
All evaluation metrics for the cross-lingual RAG paper.

RETRIEVAL METRICS:
  - Recall@K
  - Precision@K
  - MRR (Mean Reciprocal Rank)
  - NDCG@K

GENERATION METRICS:
  - BLEU-4 (sacrebleu)
  - ROUGE-L
  - BERTScore (multilingual)
  - Faithfulness score (simple lexical overlap with retrieved docs)
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# RETRIEVAL METRICS
# ══════════════════════════════════════════════════════

def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Recall@K: fraction of relevant docs found in top K."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Precision@K: fraction of top-K that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    hits = sum(1 for rid in top_k if rid in rel_set)
    return hits / k


def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> float:
    """Reciprocal Rank for a single query."""
    rel_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in rel_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """NDCG@K with binary relevance."""
    rel_set = set(relevant_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in rel_set:
            dcg += 1.0 / np.log2(i + 1)

    # Ideal DCG: all relevant docs ranked first
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    all_retrieved: List[List[str]],      # passage_ids per query
    all_relevant: List[List[str]],       # gold passage_ids per query
    k_values: List[int] = [3, 5, 10],
) -> Dict[str, float]:
    """
    Aggregate retrieval metrics over all queries.
    Returns dict: {"Recall@3": 0.xx, "Precision@5": 0.xx, "MRR": 0.xx, "NDCG@10": 0.xx, ...}
    """
    scores = defaultdict(list)

    for retrieved, relevant in zip(all_retrieved, all_relevant):
        scores["MRR"].append(reciprocal_rank(retrieved, relevant))
        for k in k_values:
            scores[f"Recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            scores[f"Precision@{k}"].append(precision_at_k(retrieved, relevant, k))
            scores[f"NDCG@{k}"].append(ndcg_at_k(retrieved, relevant, k))

    return {metric: float(np.mean(vals)) for metric, vals in scores.items()}


# ══════════════════════════════════════════════════════
# GENERATION METRICS
# ══════════════════════════════════════════════════════

def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """BLEU-4 using sacrebleu."""
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        return bleu.score / 100.0  # normalize to [0, 1]
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        return 0.0


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """ROUGE-L using rouge_score."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        return float(np.mean(scores))
    except Exception as e:
        logger.warning(f"ROUGE-L computation failed: {e}")
        return 0.0


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "ar",
) -> Dict[str, float]:
    """
    BERTScore using multilingual BERT.
    lang: "ar" for Arabic, "ms" for Malay (falls back to "others").
    """
    try:
        from bert_score import score as bert_score_fn
        model_type = "bert-base-multilingual-cased"
        P, R, F = bert_score_fn(
            predictions,
            references,
            model_type=model_type,
            lang=lang if lang in ["ar", "ms"] else "others",
            verbose=False,
            rescale_with_baseline=False,
        )
        return {
            "BERTScore_P": float(P.mean()),
            "BERTScore_R": float(R.mean()),
            "BERTScore_F": float(F.mean()),
        }
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        return {"BERTScore_P": 0.0, "BERTScore_R": 0.0, "BERTScore_F": 0.0}


def compute_faithfulness(
    predictions: List[str],
    retrieved_passages: List[List[Tuple[Dict, float]]],
) -> float:
    """
    Simple lexical faithfulness: fraction of unigrams in the generated answer
    that appear in the retrieved context.

    This is a lightweight proxy. For paper submission, supplement with
    LLM-based faithfulness (e.g., GPT-4 judge) or NLI-based scoring.
    """
    faithfulness_scores = []
    for pred, passages in zip(predictions, retrieved_passages):
        context = " ".join(p["text"] for p, _ in passages)
        context_tokens = set(context.lower().split())
        pred_tokens = pred.lower().split()
        if not pred_tokens:
            faithfulness_scores.append(0.0)
            continue
        overlap = sum(1 for t in pred_tokens if t in context_tokens)
        faithfulness_scores.append(overlap / len(pred_tokens))
    return float(np.mean(faithfulness_scores))


def compute_generation_metrics(
    predictions: List[str],
    references: List[str],
    retrieved_passages: Optional[List[List[Tuple[Dict, float]]]] = None,
    lang: str = "ar",
) -> Dict[str, float]:
    """
    Compute all generation metrics and return as a flat dict.
    """
    results = {}
    results["BLEU-4"]  = compute_bleu(predictions, references)
    results["ROUGE-L"] = compute_rouge_l(predictions, references)
    results.update(compute_bertscore(predictions, references, lang=lang))

    if retrieved_passages is not None:
        results["Faithfulness"] = compute_faithfulness(predictions, retrieved_passages)

    return results


# ══════════════════════════════════════════════════════
# COMBINED REPORT
# ══════════════════════════════════════════════════════

def full_evaluation_report(
    retrieved_ids:     List[List[str]],
    relevant_ids:      List[List[str]],
    predictions:       List[str],
    references:        List[str],
    retrieved_passages: Optional[List] = None,
    lang:              str = "ar",
    k_values:          List[int] = [3, 5, 10],
) -> Dict[str, float]:
    """
    One-call evaluation returning all 8 metrics for one experiment config.
    """
    retrieval = compute_retrieval_metrics(retrieved_ids, relevant_ids, k_values)
    generation = compute_generation_metrics(predictions, references, retrieved_passages, lang)
    return {**retrieval, **generation}
