"""
run_camel_bm25.py — CAMeL Tools morphologically-stemmed BM25 experiment.

Adds one row to Table 1: BM25+CAMeL, testing whether full morphological
stemming (vs. BM25+Norm's lightweight diacritic/alef normalization)
narrows the gap with dense retrieval further (Limitations, Section 4.2).

Usage:
    python3 run_camel_bm25.py

Output:
    results/exp1_bm25camel_ar_results.json
"""

import json, os, sys, time, math
sys.path.insert(0, os.path.dirname(__file__))

from rank_bm25 import BM25Okapi
from data.load_mkqa import load_mkqa
from data.load_wikipedia import build_corpus as load_corpus
from config import OUTPUT_DIR

from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.utils.dediac import dediac_ar

mle = MLEDisambiguator.pretrained()

def camel_stem(text: str) -> list:
    """Tokenize + morphologically stem Arabic text via CAMeL Tools MLE disambiguator."""
    if not text:
        return []
    toks = simple_word_tokenize(dediac_ar(text))
    disambig = mle.disambiguate(toks)
    stems = []
    for d in disambig:
        if d.analyses:
            lex = d.analyses[0].analysis.get('lex', d.word)
            stems.append(lex)
        else:
            stems.append(d.word)
    return stems

def compute_metrics(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    hits = [1 if pid in relevant_set else 0 for pid in retrieved_ids[:k]]
    recall = min(sum(hits), 1) / 1 if relevant_ids else 0
    prec = sum(hits) / k
    mrr = 0.0
    for i, h in enumerate(hits):
        if h:
            mrr = 1.0 / (i + 1)
            break
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, prec, mrr, ndcg

def main():
    print("=" * 60)
    print("  BM25 + CAMeL Tools Morphological Stemming Experiment")
    print("=" * 60)

    print("Loading MKQA (ar, 1000 samples)...")
    mkqa = load_mkqa(languages=["ar"], max_samples=1000)
    queries = mkqa["ar"]
    print(f"  {len(queries)} queries loaded.")

    print("Loading Arabic Wikipedia corpus...")
    passages = load_corpus("ar")
    print(f"  {len(passages)} passages loaded.")

    print("Stemming corpus with CAMeL Tools (this is the slow step)...")
    t0 = time.time()
    tokenized_corpus = []
    for i, p in enumerate(passages):
        tokenized_corpus.append(camel_stem(p["text"]))
        if (i + 1) % 5000 == 0:
            print(f"  stemmed {i+1}/{len(passages)} passages ({time.time()-t0:.0f}s elapsed)")
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  Index built in {time.time()-t0:.1f}s")

    K = 5
    results = {k: {"recall": [], "prec": [], "mrr": [], "ndcg": []} for k in [3, 5, 10]}

    print(f"Evaluating {len(queries)} queries at K={K}...")
    for i, item in enumerate(queries):
        query_stemmed = camel_stem(item["query"])
        scores = bm25.get_scores(query_stemmed)
        ranked_ids = sorted(range(len(passages)), key=lambda j: scores[j], reverse=True)[:10]
        ranked_pids = [passages[j]["passage_id"] for j in ranked_ids]

        answer = (item.get("answer") or "").strip().lower()
        relevant = [p["passage_id"] for p in passages if answer and answer in p["text"].lower()]

        for k in [3, 5, 10]:
            r, pr, m, n = compute_metrics(ranked_pids[:k], relevant, k)
            results[k]["recall"].append(r)
            results[k]["prec"].append(pr)
            results[k]["mrr"].append(m)
            results[k]["ndcg"].append(n)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(queries)}]")

    def mean(lst): return sum(lst) / len(lst) if lst else 0.0
    def ci95(lst):
        n = len(lst); p = mean(lst)
        return 1.96 * math.sqrt(p * (1 - p) / n) if n > 0 else 0.0

    agg = {k: {m: mean(results[k][m]) for m in ["recall", "prec", "mrr", "ndcg"]} for k in [3, 5, 10]}
    r5 = agg[5]["recall"]; ci_r5 = ci95(results[5]["recall"])
    mrr = agg[5]["mrr"];   ci_mrr = ci95(results[5]["mrr"])
    ndcg = agg[10]["ndcg"]; ci_ndcg = ci95(results[10]["ndcg"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = {
        "experiment": "exp1_bm25camel_ar",
        "retriever": "bm25_camel_stemmed",
        "query_lang": "ar", "corpus_lang": "ar", "top_k": 5,
        "metrics": {f"Recall@{k}": agg[k]["recall"] for k in [3,5,10]}
                  | {f"Precision@{k}": agg[k]["prec"] for k in [3,5,10]}
                  | {"MRR": mrr, "NDCG@10": ndcg},
        "ci": {"Recall@5": ci_r5, "MRR": ci_mrr, "NDCG@10": ci_ndcg}
    }
    out_path = os.path.join(OUTPUT_DIR, "exp1_bm25camel_ar_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    print("\n" + "=" * 60)
    print("  Paste this row into Table 2 after the BM25+Norm row:")
    print(f"  BM25+CAMeL & AR & {r5:.3f}$_{{\\pm{ci_r5:.3f}}}$ "
          f"& {mrr:.3f}$_{{\\pm{ci_mrr:.3f}}}$ "
          f"& {ndcg:.3f} \\\\")
    print("=" * 60)

if __name__ == "__main__":
    main()
