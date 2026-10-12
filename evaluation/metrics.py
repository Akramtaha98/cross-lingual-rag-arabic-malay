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


def hit_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Hit@K: 1.0 if at least one relevant doc appears in top K, else 0.0.

    Distinct from Recall@K, which is the FRACTION of all relevant docs
    found in top K (can be < Hit@K when a query has more than one
    relevant passage, which happens often here because relevance is
    labeled by substring match over the full 50K-passage corpus and a
    common gold-answer string can match many passages)."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return 1.0 if any(rid in top_k for rid in relevant_ids) else 0.0


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
            scores[f"Hit@{k}"].append(hit_at_k(retrieved, relevant, k))

    return {metric: float(np.mean(vals)) for metric, vals in scores.items()}


def compute_retrieval_metrics_with_per_query(
    all_retrieved: List[List[str]],
    all_relevant: List[List[str]],
    k_values: List[int] = [3, 5, 10],
):
    """Same as compute_retrieval_metrics but also returns the raw
    per-query score vectors, needed for genuine bootstrap resampling
    and paired significance tests (Section 4.2)."""
    scores = defaultdict(list)

    for retrieved, relevant in zip(all_retrieved, all_relevant):
        scores["MRR"].append(reciprocal_rank(retrieved, relevant))
        for k in k_values:
            scores[f"Recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            scores[f"Precision@{k}"].append(precision_at_k(retrieved, relevant, k))
            scores[f"NDCG@{k}"].append(ndcg_at_k(retrieved, relevant, k))
            scores[f"Hit@{k}"].append(hit_at_k(retrieved, relevant, k))

    aggregate = {metric: float(np.mean(vals)) for metric, vals in scores.items()}
    per_query = {metric: vals for metric, vals in scores.items()}
    n_relevant = [len(r) for r in all_relevant]
    return aggregate, per_query, n_relevant


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


class _UnicodeWhitespaceTokenizer:
    """Lowercase + Unicode-aware whitespace tokenizer for rouge_score.

    BUG FIX (peer review, Sept 2026): rouge_score's DefaultTokenizer applies
    re.sub(r"[^a-z0-9]+", " ", text.lower()) before splitting, which strips
    every character outside [a-z0-9] -- silently deleting all Arabic (and any
    other non-Latin-script) text and producing tokenize() -> [] for every
    Arabic string. This made ROUGE-L identically 0.0 for every Arabic row in
    every table in this study, which is a measurement artifact, not a
    genuine zero-overlap result. This tokenizer just lowercases and splits on
    whitespace, so non-Latin scripts are preserved.
    """
    def tokenize(self, text):
        return text.lower().split()


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """ROUGE-L using rouge_score, with a script-agnostic tokenizer (see
    _UnicodeWhitespaceTokenizer above) so non-Latin scripts such as Arabic
    are not silently zeroed out before scoring."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=False, tokenizer=_UnicodeWhitespaceTokenizer()
        )
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
