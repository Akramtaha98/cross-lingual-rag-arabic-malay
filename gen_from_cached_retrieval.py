#!/usr/bin/env python3
"""Fast(ish) step 2: given results/table9_matched_retrieved_{lang}.json (from
precompute_matched_retrieval.py), run mT5-small generation for ONE variant
(base or lora), checkpointing predictions to a .jsonl as it goes so it can
be resumed across multiple short calls (fixes the earlier outputs[:10]
truncation bug — final output saves ALL samples so a real n=50 judge is
possible).

Usage (repeat until "ALL DONE" prints, then it also writes the final
results JSON with scores):
    python3 gen_from_cached_retrieval.py --lang ar --variant base --time_budget 150
"""
import argparse
import json
import sys
import time

sys.path.insert(0, ".")
from generation.mt5_generator import MT5Generator  # noqa: E402
from evaluation.metrics import compute_bertscore, compute_bleu, compute_rouge_l, compute_faithfulness  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["ar", "ms"])
    ap.add_argument("--variant", required=True, choices=["base", "lora"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--time_budget", type=int, default=150, help="seconds before stopping to checkpoint")
    args = ap.parse_args()

    cache = json.load(open(f"results/table9_matched_retrieved_{args.lang}.json", encoding="utf-8"))
    samples = cache["samples"]
    retrieved_all = cache["retrieved"]
    n_total = len(samples)

    ckpt_path = f"results/_ckpt_matched_{args.variant}_{args.lang}.jsonl"
    done = []
    try:
        with open(ckpt_path, encoding="utf-8") as f:
            for line in f:
                done.append(json.loads(line))
    except FileNotFoundError:
        pass
    start_idx = len(done)
    print(f"Loaded {n_total} cached samples; {start_idx} already generated.", flush=True)

    if start_idx >= n_total:
        print("All predictions already generated, skipping to scoring.", flush=True)
    else:
        use_lora = args.variant == "lora"
        gen = MT5Generator(use_lora=use_lora)
        print(f"mT5-small loaded (use_lora={use_lora})", flush=True)

        t0 = time.time()
        ckpt_f = open(ckpt_path, "a", encoding="utf-8")
        i = start_idx
        while i < n_total:
            s, retrieved = samples[i], retrieved_all[i]
            pred = gen.generate(s["query"], retrieved, top_k=args.k)
            ckpt_f.write(json.dumps({"i": i, "answer": pred}, ensure_ascii=False) + "\n")
            ckpt_f.flush()
            i += 1
            if i % 25 == 0:
                print(f"  generated {i}/{n_total}", flush=True)
            if time.time() - t0 > args.time_budget:
                print(f"Time budget hit at {i}/{n_total} — rerun the same command to resume.", flush=True)
                ckpt_f.close()
                return
        ckpt_f.close()

    # All done -> score and write final results file
    done = []
    with open(ckpt_path, encoding="utf-8") as f:
        for line in f:
            done.append(json.loads(line))
    done.sort(key=lambda d: d["i"])
    preds = [d["answer"] for d in done]
    assert len(preds) == n_total, f"expected {n_total} preds, got {len(preds)}"

    refs = [s["reference"] for s in samples]
    bleu = compute_bleu(preds, refs)
    rouge = compute_rouge_l(preds, refs)
    bert = compute_bertscore(preds, refs, lang=args.lang)
    faith = compute_faithfulness(preds, retrieved_all)

    out = {
        "config": {"generator": f"mt5-small+{args.variant}", "retriever": "me5large",
                   "lang": args.lang, "n": n_total, "k": args.k},
        "metrics": {**bert, "BLEU-4": bleu, "ROUGE-L": rouge, "Faithfulness_lex_mean": faith},
        "samples": [{"query": s["query"], "reference": s["reference"], "answer": p, "lang": args.lang}
                    for s, p in zip(samples, preds)],
    }
    out_path = f"results/table9_matched_me5large_mt5_{args.variant}_{args.lang}_results.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"ALL DONE {args.variant}/{args.lang}: BERTScore-F={bert['BERTScore_F']:.4f} "
          f"BLEU-4={bleu:.4f} Faith={faith:.4f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
