"""
run_matched_retriever_gen.py

Peer-review item 9 (Table 9 retriever mismatch): the strong-generator
comparison (run_strong_generator.py, results/strong_gen_qwen2.5_7b_*)
used Qwen2.5-7B-Instruct with mE5-large retrieval, while the mT5-small
rows in Table 9 used LaBSE retrieval (the EXP 3 configuration) at a
different sample size (n=1,000 vs n=200). That confounds generator
capacity with retriever quality and sample size.

This script reruns mT5-small (base and LoRA) with the SAME retriever
(mE5-large), the SAME first-200-MKQA-item sample, and the SAME top_k=5
as the existing Qwen run, so Table 9 gets one genuinely matched
(retriever, n) comparison: mT5-base vs mT5+LoRA vs Qwen2.5-7B, all under
mE5-large retrieval, n=200, K=5.
"""
import argparse
import gzip
import json
import logging

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from generation.mt5_generator import MT5Generator
from evaluation.metrics import compute_bertscore, compute_bleu, compute_rouge_l, compute_faithfulness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_mkqa_first_n(path, lang, n=200):
    samples = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            query = item.get("queries", {}).get(lang) or ""
            ans_list = item.get("answers", {}).get(lang, [])
            if isinstance(ans_list, list) and ans_list:
                answer_text = ans_list[0].get("text", "") if isinstance(ans_list[0], dict) else str(ans_list[0])
            else:
                answer_text = ""
            if query and answer_text:
                samples.append({"query": query, "reference": answer_text, "lang": lang})
            if len(samples) >= n:
                break
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["ar", "ms"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--variant", choices=["base", "lora", "both"], default="both")
    args = ap.parse_args()
    lang = args.lang

    samples = load_mkqa_first_n("indexes/mkqa.jsonl.gz", lang, args.n)
    logger.info(f"[{lang}] loaded {len(samples)} matched samples (same first-{args.n} MKQA items as the Qwen run).")

    index = faiss.read_index(f"indexes/me5large_{lang}.faiss")
    passages = json.load(open(f"indexes/me5large_{lang}_passages.json", encoding="utf-8"))
    logger.info(f"Loaded mE5-large FAISS index ({index.ntotal} vectors) + {len(passages)} passages")

    encoder = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")

    def retrieve(query, k):
        vec = encoder.encode([f"query: {query}"], normalize_embeddings=True)
        scores, idxs = index.search(np.asarray(vec, dtype="float32"), k)
        return [(passages[i], float(s)) for s, i in zip(scores[0], idxs[0]) if 0 <= i < len(passages)]

    logger.info("Retrieving mE5-large top-%d passages for all samples...", args.k)
    retrieved_all = [retrieve(s["query"], args.k) for s in samples]

    variants = [("base", False), ("lora", True)]
    if args.variant != "both":
        variants = [(v, ul) for v, ul in variants if v == args.variant]
    for variant, use_lora in variants:
        logger.info(f"Loading mT5-small (use_lora={use_lora})...")
        gen = MT5Generator(use_lora=use_lora)

        preds = []
        for s, retrieved in zip(samples, retrieved_all):
            pred = gen.generate(s["query"], retrieved, top_k=args.k)
            preds.append(pred)

        refs = [s["reference"] for s in samples]
        bleu = compute_bleu(preds, refs)
        rouge = compute_rouge_l(preds, refs)
        bert = compute_bertscore(preds, refs, lang=lang)
        faith = compute_faithfulness(preds, retrieved_all)

        out = {
            "config": {"generator": f"mt5-small+{'lora' if use_lora else 'base'}",
                       "retriever": "me5large", "lang": lang, "n": len(samples), "k": args.k},
            "metrics": {**bert, "BLEU-4": bleu, "ROUGE-L": rouge, "Faithfulness_lex_mean": faith},
            "samples": [{"query": s["query"], "reference": s["reference"], "answer": p, "lang": lang}
                        for s, p in zip(samples, preds)],
        }
        out_path = f"results/table9_matched_me5large_mt5_{variant}_{lang}_results.json"
        json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        logger.info(f"{variant}/{lang}: BERTScore-F={bert['BERTScore_F']:.4f} BLEU-4={bleu:.4f} Faith={faith:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
