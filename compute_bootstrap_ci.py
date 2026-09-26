"""
compute_bootstrap_ci.py
=======================
Computes 95% confidence intervals for all key metrics across all experiments.

Since the result JSONs store only 10 sample-level entries (for inspection) but
record full-scale metrics over n=1,000 queries, we use the CLT-based interval:

  For proportion metrics (Recall@K, Precision@K):
      SE = sqrt(p*(1-p)/n)           [exact normal approximation for binomials]
      CI = p ± 1.96 * SE

  For bounded-but-non-proportion metrics (MRR, NDCG, BERTScore-F):
      Per-query MRR ∈ [0,1], NDCG ∈ [0,1], BERTScore-F ∈ [0,1]
      Hoeffding-based conservative SE = (b-a)/(2*sqrt(n)) where range [a,b]
      For MRR/NDCG: SE_max = 1/(2*sqrt(n)) ≈ 0.0158 at n=1000
      We use SE = sqrt(mu*(1-mu)/n) as a tighter approximation.

This produces honest, defensible CIs for the paper tables.
Output: ci_results.json + prints a LaTeX-ready table fragment.

Usage:
    cd cross_lingual_rag
    python compute_bootstrap_ci.py
"""

import json
import math
import os
from pathlib import Path

RESULTS_DIR = Path("./results")
N = 1000   # queries per experiment


def ci_proportion(p, n, z=1.96):
    """95% CI for a proportion using normal approximation (CLT)."""
    se = math.sqrt(p * (1 - p) / n) if p > 0 else 0.0
    return p - z * se, p + z * se


def ci_bounded(mu, n, lo=0.0, hi=1.0, z=1.96):
    """
    95% CI for a bounded metric in [lo, hi] using Hoeffding SE as ceiling,
    tightened by CLT approximation sqrt(mu*(1-mu)/n).
    """
    # CLT approximation (treats mu like a proportion – tighter than Hoeffding)
    se_clt = math.sqrt(mu * (1 - mu) / n) if 0 < mu < 1 else 0.0
    # Hoeffding conservative upper bound
    se_hoef = (hi - lo) / (2 * math.sqrt(n))
    se = min(se_clt if se_clt > 0 else se_hoef, se_hoef)
    return mu - z * se, mu + z * se


def fmt(val, lo, hi, decimals=4):
    """Format 'val ± half_width' string."""
    hw = (hi - lo) / 2
    fmt_str = f"{{:.{decimals}f}}"
    return f"{fmt_str.format(val)} ± {fmt_str.format(hw)}"


def load_metrics(exp_name):
    path = RESULTS_DIR / f"{exp_name}_results.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metrics", {})


def _get(m, *keys):
    for k in keys:
        if k in m:
            return m[k]
    return None


# ── Experiments to process ───────────────────────────────────────────────────

EXPERIMENTS = {
    # EXP1 retriever comparison
    "exp1_bm25_ar":     ("BM25",      "AR"),
    "exp1_bm25_ms":     ("BM25",      "MS"),
    "exp1_labse_ar":    ("LaBSE",     "AR"),
    "exp1_labse_ms":    ("LaBSE",     "MS"),
    "exp1_me5_ar":      ("mE5-base",  "AR"),
    "exp1_me5_ms":      ("mE5-base",  "MS"),
    "exp1_me5large_ar": ("mE5-large", "AR"),
    "exp1_me5large_ms": ("mE5-large", "MS"),
    "exp1_bgem3_ar":    ("BGE-M3",    "AR"),
    "exp1_bgem3_ms":    ("BGE-M3",    "MS"),
    # EXP2 language direction
    "exp2_ar_mono":     ("Mono",       "AR"),
    "exp2_ar_cross":    ("Cross",      "AR→MS"),
    "exp2_ms_mono":     ("Mono",       "MS"),
    "exp2_ms_cross":    ("Cross",      "MS→AR"),
    # EXP3 generator
    "exp3_base_ar":     ("mT5-base",   "AR"),
    "exp3_base_ms":     ("mT5-base",   "MS"),
    "exp3_lora_ar":     ("mT5+LoRA",   "AR"),
    "exp3_lora_ms":     ("mT5+LoRA",   "MS"),
    # EXP4 top-K
    "exp4_k3":          ("K=3",        "AR"),
    "exp4_k5":          ("K=5",        "AR"),
    "exp4_k10":         ("K=10",       "AR"),
    # EXP5 transfer
    "exp5_ar_to_ms":    ("AR→MS",      "cross"),
    "exp5_ms_to_ar":    ("MS→AR",      "cross"),
}


def compute_ci_for_exp(exp_name):
    m = load_metrics(exp_name)
    if not m:
        return {}

    results = {"n": N, "metrics": {}}

    # ── Retrieval metrics ────────────────────────────────────────────────────
    for key, col in [
        ("Recall@5",    ("Recall@5", "recall_at_k")),
        ("Recall@3",    ("Recall@3",)),
        ("Recall@10",   ("Recall@10",)),
        ("Precision@5", ("Precision@5", "precision_at_k")),
        ("Precision@3", ("Precision@3",)),
        ("Precision@10",("Precision@10",)),
    ]:
        val = _get(m, *col)
        if val is not None:
            lo, hi = ci_proportion(val, N)
            results["metrics"][key] = {"mean": val, "lo": lo, "hi": hi,
                                        "fmt": fmt(val, lo, hi)}

    for key, col in [
        ("MRR",     ("MRR", "mrr")),
        ("NDCG@5",  ("NDCG@5", "ndcg_at_k")),
        ("NDCG@10", ("NDCG@10",)),
    ]:
        val = _get(m, *col)
        if val is not None:
            lo, hi = ci_bounded(val, N)
            results["metrics"][key] = {"mean": val, "lo": lo, "hi": hi,
                                        "fmt": fmt(val, lo, hi)}

    # ── Generation metrics ───────────────────────────────────────────────────
    for key, col in [
        ("BLEU-4",       ("BLEU-4", "bleu4")),
        ("ROUGE-L",      ("ROUGE-L", "rougeL")),
        ("BERTScore_F",  ("BERTScore_F", "bertscore_f")),
        ("Faithfulness", ("Faithfulness", "faithfulness")),
    ]:
        val = _get(m, *col)
        if val is not None:
            lo, hi = ci_bounded(val, N)
            results["metrics"][key] = {"mean": val, "lo": lo, "hi": hi,
                                        "fmt": fmt(val, lo, hi)}

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_ci = {}
    for exp_name in EXPERIMENTS:
        ci = compute_ci_for_exp(exp_name)
        if ci:
            all_ci[exp_name] = ci
            label, lang = EXPERIMENTS[exp_name]
            m = ci["metrics"]
            r5   = m.get("Recall@5",   {}).get("fmt", "N/A")
            mrr  = m.get("MRR",        {}).get("fmt", "N/A")
            ndcg = m.get("NDCG@5",     {}) or m.get("NDCG@10", {})
            ndcg_fmt = ndcg.get("fmt", "N/A") if ndcg else "N/A"
            bsf  = m.get("BERTScore_F",{}).get("fmt", "N/A")
            print(f"{exp_name:<22} | {label:<10} {lang:<6} | "
                  f"R@5={r5}  MRR={mrr}  NDCG={ndcg_fmt}  BSF={bsf}")

    out = RESULTS_DIR / "ci_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_ci, f, indent=2)
    print(f"\nSaved → {out}")

    # ── Print LaTeX-friendly table rows for key EXP1 results ────────────────
    print("\n--- LaTeX EXP1 Table rows (Recall@5 / MRR / NDCG@5) ---")
    exp1_order = [
        ("exp1_bm25_ar",     "BM25",      "AR"),
        ("exp1_bm25_ms",     "BM25",      "MS"),
        ("exp1_labse_ar",    "LaBSE",     "AR"),
        ("exp1_labse_ms",    "LaBSE",     "MS"),
        ("exp1_me5_ar",      "mE5-base",  "AR"),
        ("exp1_me5_ms",      "mE5-base",  "MS"),
        ("exp1_me5large_ar", "mE5-large", "AR"),
        ("exp1_me5large_ms", "mE5-large", "MS"),
        ("exp1_bgem3_ar",    "BGE-M3",    "AR"),
        ("exp1_bgem3_ms",    "BGE-M3",    "MS"),
    ]
    print("Retriever & Lang & Recall@5 & MRR & NDCG@5 \\\\")
    for exp_name, label, lang in exp1_order:
        ci = all_ci.get(exp_name, {}).get("metrics", {})
        r5  = ci.get("Recall@5",  {}).get("fmt", "—")
        mrr = ci.get("MRR",       {}).get("fmt", "—")
        nd  = ci.get("NDCG@5",    {}) or ci.get("NDCG@10", {})
        nd_fmt = nd.get("fmt", "—") if nd else "—"
        print(f"{label} & {lang} & {r5} & {mrr} & {nd_fmt} \\\\")


if __name__ == "__main__":
    main()
