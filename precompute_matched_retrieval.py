#!/usr/bin/env python3
"""One-time (per language) retrieval precompute for the matched-retriever
Table 9 rerun, so the slow encoder load doesn't have to happen on every call.

Writes results/table9_matched_retrieved_{lang}.json:
  {"samples": [...], "retrieved": [[[passage, score], ...], ...]}
"""
import argparse
import json
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, ".")
from run_matched_retriever_gen import load_mkqa_first_n  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["ar", "ms"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    samples = load_mkqa_first_n("indexes/mkqa.jsonl.gz", args.lang, args.n)
    print(f"[{args.lang}] loaded {len(samples)} samples", flush=True)

    index = faiss.read_index(f"indexes/me5large_{args.lang}.faiss")
    passages = json.load(open(f"indexes/me5large_{args.lang}_passages.json", encoding="utf-8"))
    print(f"Loaded FAISS index ({index.ntotal} vectors) + {len(passages)} passages", flush=True)

    encoder = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
    print("Encoder loaded, retrieving...", flush=True)

    retrieved_all = []
    for s in samples:
        vec = encoder.encode([f"query: {s['query']}"], normalize_embeddings=True)
        scores, idxs = index.search(np.asarray(vec, dtype="float32"), args.k)
        retrieved_all.append([[passages[i], float(sc)] for sc, i in zip(scores[0], idxs[0]) if 0 <= i < len(passages)])

    out = {"samples": samples, "retrieved": retrieved_all}
    out_path = f"results/table9_matched_retrieved_{args.lang}.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
