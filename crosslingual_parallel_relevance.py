#!/usr/bin/env python3
"""Cross-lingual retrieval re-evaluated with CORPUS-LANGUAGE gold answers.
Earlier EXP2/EXP5 labelled a passage relevant if it contained the QUERY-language gold answer string,
which (e.g. Arabic string vs Malay corpus) can only match digits/shared names. Here relevance uses the
parallel MKQA answer in the corpus language. Resumable: rerun until 'DONE'.
usage: python3 crosslingual_parallel_relevance.py --ret me5large --qlang ar --budget 150
"""
import argparse, gzip, json, os, sys, time
import numpy as np, faiss
from sentence_transformers import SentenceTransformer
ap = argparse.ArgumentParser()
ap.add_argument("--ret", required=True, choices=["labse", "me5large"])
ap.add_argument("--qlang", required=True, choices=["ar", "ms"])
ap.add_argument("--budget", type=int, default=150)
a = ap.parse_args()
clang = "ms" if a.qlang == "ar" else "ar"
N, K = 1000, 10
rows = []
for i, l in enumerate(gzip.open("indexes/mkqa.jsonl.gz", "rt", encoding="utf-8")):
    rows.append(json.loads(l))
    if i + 1 >= N: break
def ans(r, lang):
    x = r.get("answers", {}).get(lang) or []
    return (x[0].get("text") or "").strip().lower() if x and isinstance(x[0], dict) else ""
queries = [r.get("queries", {}).get(a.qlang) or "" for r in rows]
tag = f"{a.ret}_{a.qlang}to{clang}"
os.makedirs("results", exist_ok=True)
embp = f"results/_qemb_{tag}.npy"
embs = list(np.load(embp)) if os.path.exists(embp) else []
if len(embs) < N:
    model = SentenceTransformer("intfloat/multilingual-e5-large" if a.ret == "me5large" else "sentence-transformers/LaBSE", device=__import__("os").environ.get("DEV","cpu"))
    pref = "query: " if a.ret == "me5large" else ""
    t0 = time.time()
    while len(embs) < N and time.time() - t0 < a.budget:
        i = len(embs); batch = queries[i:i+50]
        v = model.encode([pref + q for q in batch], normalize_embeddings=True)
        embs.extend(list(v)); np.save(embp, np.array(embs))
    if len(embs) < N:
        print(f"encoded {len(embs)}/{N}; rerun to resume"); sys.exit(0)
index = faiss.read_index(f"indexes/{a.ret}_{clang}.faiss")
passages = json.load(open(f"indexes/{a.ret}_{clang}_passages.json", encoding="utf-8"))
texts = [(p["text"] if isinstance(p, dict) else p).lower() for p in passages]
E=np.array(embs, dtype="float32"); _, I = index.search(E, K)
hit5, rec5, mrr, nrel = [], [], [], []
for q in range(N):
    g = ans(rows[q], clang)
    if not queries[q] or not g:
        hit5.append(0.0); rec5.append(0.0); mrr.append(0.0); nrel.append(0); continue
    rel = {j for j, t in enumerate(texts) if g in t}
    nrel.append(len(rel)); top = list(I[q])
    h = [1 if j in rel else 0 for j in top]
    hit5.append(float(any(h[:5]))); rec5.append(sum(h[:5]) / len(rel) if rel else 0.0)
    mrr.append(next((1/(r+1) for r, x in enumerate(h) if x), 0.0))
rng = np.random.default_rng(42)
def ci(v):
    v = np.array(v); b = [v[rng.integers(0, N, N)].mean() for _ in range(2000)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
out = {"tag": tag, "n": N, "mean_relevant": float(np.mean([x for x in nrel if x])),
       "Hit@5": ci(hit5), "R@5": ci(rec5), "MRR": ci(mrr),
       "per_query": {"hit5": hit5, "rec5": rec5, "mrr": mrr}}
json.dump(out, open(f"results/xl_parallel_{tag}.json", "w"))
print("DONE", tag, {k: [round(x, 4) for x in out[k]] for k in ("Hit@5", "R@5", "MRR")}, "mean_rel", round(out["mean_relevant"], 1))
