"""
run_all_experiments.py
Master script: run all 5 experiments and write results to CSV + JSON.

Usage:
    python run_all_experiments.py           # full evaluation (1,000 samples/lang)
    python run_all_experiments.py --dev     # tiny dev set (~10 samples) for code checks
    python run_all_experiments.py --exp exp1  # only exp1_* configs
    python run_all_experiments.py --rebuild_index  # force rebuild FAISS indexes

Modes:
    default (--full)  → 1,000 samples per language, fixed generator, valid metrics
    --dev             → 10 samples only; for pipeline verification ONLY;
                        metrics from --dev are NOT publication-quality

Results saved to: ./results/<exp_name>_results.json
Summary table:    ./results/summary_table.csv
"""

import os
import json
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from config import EXPERIMENTS, OUTPUT_DIR, MKQA_MAX_SAMPLES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


FULL_SAMPLES = 1000   # minimum for publication-quality results
DEV_SAMPLES  = 10    # pipeline smoke-test only


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default=None, help="Filter experiments by name prefix")
    p.add_argument(
        "--dev",
        action="store_true",
        help=(
            f"Tiny dev mode: {DEV_SAMPLES} samples per language. "
            "For pipeline verification ONLY — NOT publication-quality."
        ),
    )
    p.add_argument(
        "--samples",
        type=int,
        default=None,
        help=f"Override number of MKQA samples per language (default: {FULL_SAMPLES}).",
    )
    p.add_argument("--rebuild_index", action="store_true", help="Force rebuild FAISS indexes")
    p.add_argument("--top_k", type=int, default=None, help="Override top_k for all experiments")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        dest="skip_existing",
        help="Skip experiments that already have a results JSON (safe resume after crash).",
    )
    return p.parse_args()


def get_relevant_passage_ids(query_item: dict, passages: list) -> list:
    """
    Heuristic: a passage is 'relevant' if it contains the gold answer string.
    In a real evaluation, use annotated passage-level relevance labels.
    This proxy is standard in open-domain QA when only answer strings are available.
    """
    answer = query_item.get("answer", "").strip().lower()
    if not answer:
        return []
    return [
        p["passage_id"] for p in passages
        if answer in p["text"].lower()
    ]


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────
    from data.load_mkqa import load_mkqa, split_mkqa
    from data.load_wikipedia import load_both_corpora

    if args.dev:
        max_samples = DEV_SAMPLES
        logger.warning(
            f"⚠  DEV MODE: only {DEV_SAMPLES} samples — results are NOT publication-quality."
        )
    elif args.samples:
        max_samples = args.samples
    else:
        max_samples = FULL_SAMPLES
        logger.info(f"FULL MODE: {FULL_SAMPLES} samples per language (publication quality).")
    logger.info(f"Loading MKQA (max_samples={max_samples}) …")
    mkqa_data = load_mkqa(max_samples=max_samples)
    mkqa_splits = split_mkqa(mkqa_data)

    logger.info("Loading Wikipedia corpora …")
    corpora = load_both_corpora()

    # ── Select experiments ─────────────────────────────────
    configs = EXPERIMENTS
    if args.exp:
        configs = [c for c in configs if c.name.startswith(args.exp)]
    if args.top_k:
        for c in configs:
            c.top_k = args.top_k

    logger.info(f"Running {len(configs)} experiment configurations …\n")

    all_rows = []

    for cfg in configs:
        # ── Skip already-completed experiments ─────────────────
        out_path = os.path.join(OUTPUT_DIR, f"{cfg.name}_results.json")
        if args.skip_existing and os.path.exists(out_path):
            logger.info(f"Skipping {cfg.name} (results already exist: {out_path})")
            # Still load for summary table
            with open(out_path, encoding="utf-8") as f:
                import json as _json
                saved = _json.load(f)
            row = {"experiment": cfg.name, **cfg.__dict__, **saved.get("metrics", {})}
            all_rows.append(row)
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"EXP: {cfg.name}")
        logger.info(f"  retriever={cfg.retriever}  query_lang={cfg.query_lang}  "
                    f"corpus_lang={cfg.corpus_lang}  top_k={cfg.top_k}  lora={cfg.use_lora}")
        logger.info(f"{'='*60}")

        # Build pipeline
        from pipeline.rag_pipeline import build_pipeline
        pipeline = build_pipeline(
            config=cfg,
            corpora=corpora,
            force_rebuild_index=args.rebuild_index,
        )

        # Select test set for the query language
        test_items = mkqa_splits[cfg.query_lang]["test"]
        queries    = [item["query"]  for item in test_items]
        references = [item["answer"] for item in test_items]

        # Determine corpus passages (for relevance scoring)
        if cfg.corpus_lang == "both":
            corpus_passages = corpora["ar"] + corpora["ms"]
        else:
            corpus_passages = corpora[cfg.corpus_lang]

        # Run pipeline
        logger.info(f"Running {len(queries)} queries …")
        results = pipeline.run_batch(queries, references)

        # ── Evaluate ────────────────────────────────────────
        from evaluation.metrics import full_evaluation_report

        all_retrieved_ids = [
            [p["passage_id"] for p, _ in r["passages"]]
            for r in results
        ]
        all_relevant_ids = [
            get_relevant_passage_ids(item, corpus_passages)
            for item in test_items
        ]
        predictions = [r["answer"] for r in results]

        metrics = full_evaluation_report(
            retrieved_ids=all_retrieved_ids,
            relevant_ids=all_relevant_ids,
            predictions=predictions,
            references=references,
            retrieved_passages=[r["passages"] for r in results],
            lang=cfg.query_lang,
        )

        # Log metrics
        logger.info("Results:")
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k:<20} {v:.4f}")

        # Save per-experiment results
        exp_output = {
            "config":    cfg.__dict__,
            "metrics":   metrics,
            "metadata":  {"n_queries": len(queries)},   # actual count (not len of samples list)
            "samples":   results[:10],                   # save 10 sample outputs for inspection
            "timestamp": datetime.now().isoformat(),
        }
        out_path = os.path.join(OUTPUT_DIR, f"{cfg.name}_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(exp_output, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved → {out_path}")

        # Collect for summary table
        row = {"experiment": cfg.name, **cfg.__dict__, **metrics}
        all_rows.append(row)

    # ── Summary table ───────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSummary table saved → {csv_path}")

    # Pretty-print key metrics
    key_cols = ["experiment", "BLEU-4", "ROUGE-L", "BERTScore_F",
                "Faithfulness", "Recall@5", "MRR"]
    key_cols = [c for c in key_cols if c in df.columns]
    print("\n" + df[key_cols].to_string(index=False))


if __name__ == "__main__":
    main()
