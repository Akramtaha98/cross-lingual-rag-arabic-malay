"""
run_normalized_bm25.py — Arabic morphology-normalized BM25 experiment.

Adds one row to Table 1 in the paper: BM25+Norm, showing whether
removing diacritics and normalising alef variants improves Arabic retrieval.

Usage:
    python3 run_normalized_bm25.py

Output:
    results/exp1_bm25norm_ar_results.json
    Prints ready-to-paste LaTeX row for Table 1.

Normalisation steps (pyarabic):
    1. Remove diacritics (tashkeel/harakat)
    2. Normalise alef variants (آ أ إ → ا)
    3. Normalise teh marbuta (ة → ه)
    4. Normalise waw/yeh variants (ؤ → و, ئ → ي)
"""

import json, os, sys, time, math, re
sys.path.insert(0, os.path.dirname(__file__))

import pyarabic.araby as araby
from rank_bm25 import BM25Okapi
from data.load_mkqa import load_mkqa
from data.load_wikipedia import build_corpus as load_corpus
from config import OUTPUT_DIR

# ── Arabic normalisation ──────────────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    """Apply morphological normalization for Arabic text."""
    if not text:
        return text
    text = araby.strip_tashkeel(text)          # remove diacritics
    text = araby.normalize_hamza(text)          # normalize hamza forms
    text = araby.normalize_alef(text)           # normalize alef variants
    # Belt-and-braces: catch any remaining alef variants
    text = re.sub(r'[آأإ]', 'ا', text)
    # Normalize teh marbuta → heh
    text = re.sub(r'ة', 'ه', text)
    # Normalize waw with hamza → waw
    text = re.sub(r'ؤ', 'و', text)
    # Normalize yeh with hamza → yeh
    text = re.sub(r'ئ', 'ي', text)
    return text.strip()

def tokenize(text: str) -> list:
    return normalize_arabic(text).split()

# ── Retrieval metrics ─────────────────────────────────────────────────────────

def compute_metrics(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    hits = [1 if pid in relevant_set else 0 for pid in retrieved_ids[:k]]
    recall   = min(sum(hits), 1) / 1 if relevant_ids else 0  # binary: found/not
    prec     = sum(hits) / k
    mrr      = 0.0
    for i, h in enumerate(hits):
        if h:
            mrr = 1.0 / (i + 1)
            break
    # NDCG@k
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, prec, mrr, ndcg

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BM25 + Arabic Morphology Normalization Experiment")
    print("=" * 60)

    # Load data
    print("Loading MKQA (ar, 1000 samples)...")
    mkqa = load_mkqa(languages=["ar"], max_samples=1000)
    queries = mkqa["ar"]
    print(f"  {len(queries)} queries loaded.")

    print("Loading Arabic Wikipedia corpus...")
    passages = load_corpus("ar")
    print(f"  {len(passages)} passages loaded.")

    # Build normalized BM25 index
    print("Building BM25 index with normalization...")
    t0 = time.time()
    tokenized_corpus = [tokenize(p["text"]) for p in passages]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  Index built in {time.time()-t0:.1f}s")

    # Evaluate
    K = 5
    results = {k: {"recall": [], "prec": [], "mrr": [], "ndcg": []}
               for k in [3, 5, 10]}

    print(f"Evaluating {len(queries)} queries at K={K}...")
    for i, item in enumerate(queries):
        query_norm = tokenize(item["query"])
        scores = bm25.get_scores(query_norm)
        ranked_ids = sorted(range(len(passages)),
                            key=lambda j: scores[j], reverse=True)[:10]
        ranked_pids = [passages[j]["passage_id"] for j in ranked_ids]

        # Heuristic relevance: answer string in passage text
        answer = (item.get("answer") or "").strip().lower()
        relevant = [p["passage_id"] for p in passages
                    if answer and normalize_arabic(answer) in
                    normalize_arabic(p["text"]).lower()]

        for k in [3, 5, 10]:
            r, p, m, n = compute_metrics(ranked_pids[:k], relevant, k)
            results[k]["recall"].append(r)
            results[k]["prec"].append(p)
            results[k]["mrr"].append(m)
            results[k]["ndcg"].append(n)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(queries)}]")

    # Aggregate
    def mean(lst): return sum(lst) / len(lst) if lst else 0.0
    def ci95(lst):
        n = len(lst); p = mean(lst)
        return 1.96 * math.sqrt(p * (1 - p) / n) if n > 0 else 0.0

    agg = {}
    for k in [3, 5, 10]:
        agg[k] = {m: mean(results[k][m]) for m in ["recall", "prec", "mrr", "ndcg"]}

    r5  = agg[5]["recall"];  ci_r5  = ci95(results[5]["recall"])
    mrr = agg[5]["mrr"];     ci_mrr = ci95(results[5]["mrr"])
    ndcg= agg[10]["ndcg"];   ci_ndcg= ci95(results[10]["ndcg"])

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = {
        "experiment": "exp1_bm25norm_ar",
        "retriever": "bm25_normalized",
        "query_lang": "ar", "corpus_lang": "ar", "top_k": 5,
        "metrics": {
            f"Recall@{k}":    agg[k]["recall"] for k in [3,5,10]
        } | {
            f"Precision@{k}": agg[k]["prec"]   for k in [3,5,10]
        } | {
            f"MRR":           mrr,
            f"NDCG@10":       ndcg,
        },
        "ci": {"Recall@5": ci_r5, "MRR": ci_mrr, "NDCG@10": ci_ndcg}
    }
    out_path = os.path.join(OUTPUT_DIR, "exp1_bm25norm_ar_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Print LaTeX row
    print("\n" + "=" * 60)
    print("  Paste this row into Table 1 after the BM25 row:")
    print(f"  BM25+Norm & AR & {r5:.3f}$_{{\\pm{ci_r5:.3f}}}$ "
          f"& {mrr:.3f}$_{{\\pm{ci_mrr:.3f}}}$ "
          f"& {ndcg:.3f} \\\\")
    print("=" * 60)

if __name__ == "__main__":
    main()
