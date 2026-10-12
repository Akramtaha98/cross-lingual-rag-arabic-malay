"""
run_oracle_context.py

Peer-review item 7: substantiate the "retrieval is the bottleneck, not
generation" claim with an oracle-context experiment, not just an
inference from indirect numbers.

Two conditions per language, both using the corrected mT5-small+LoRA
generator (same checkpoint as Table 4 / Section 3.3):
  - "oracle": context is a synthetic passage built directly from the gold
    answer string ("The answer is: <gold answer>."), i.e. the best-case
    context a retriever could ever hand the generator.
  - "closed_book": no retrieved context at all (empty context string).

If Adequacy/BERTScore-F jump sharply under "oracle" relative to the
retrieval-grounded numbers in Table 4, that is direct evidence retrieval
coverage -- not generator capacity -- is the binding constraint. If they
do NOT jump, the generator itself is also a bottleneck and the paper's
central claim needs qualifying.

Usage:
    python run_oracle_context.py --lang ar
    python run_oracle_context.py --lang ms
"""
import argparse
import gzip
import json
import logging

from generation.mt5_generator import MT5Generator
from evaluation.metrics import compute_bertscore, compute_bleu, compute_rouge_l

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_mkqa_first_n(path, n=1000):
    items = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            items.append(json.loads(line))
    return items


def gold_answer_string(item, lang):
    ans_list = item.get("answers", {}).get(lang, [])
    for a in ans_list:
        if a.get("text"):
            return a["text"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["ar", "ms"])
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    items = load_mkqa_first_n("indexes/mkqa.jsonl.gz", args.n)
    lang = args.lang

    queries, references, has_answer = [], [], []
    for item in items:
        q = item["queries"].get(lang)
        ans = gold_answer_string(item, lang)
        queries.append(q)
        references.append(ans if ans else "")
        has_answer.append(bool(ans))

    logger.info(f"Loaded {len(queries)} {lang} queries; {sum(has_answer)} have a gold answer string.")

    gen = MT5Generator(use_lora=True)

    for condition in ["oracle", "closed_book"]:
        predictions = []
        for q, ans, had in zip(queries, references, has_answer):
            if condition == "oracle" and had:
                oracle_text = f"The answer is: {ans}." if lang != "ar" else f"الإجابة هي: {ans}."
                passages = [({"text": oracle_text, "passage_id": "oracle_0"}, 1.0)]
            elif condition == "oracle" and not had:
                # No gold answer exists (unanswerable item) -- oracle context is empty,
                # matching closed_book for this subset; this is expected, not a bug.
                passages = []
            else:  # closed_book
                passages = []
            pred = gen.generate(q, passages, top_k=1)
            predictions.append(pred)

        # Only score items that actually have a gold reference string.
        scored_preds = [p for p, h in zip(predictions, has_answer) if h]
        scored_refs = [r for r, h in zip(references, has_answer) if h]

        bleu = compute_bleu(scored_preds, scored_refs)
        rouge = compute_rouge_l(scored_preds, scored_refs)
        bert = compute_bertscore(scored_preds, scored_refs, lang=lang)

        out = {
            "condition": condition,
            "lang": lang,
            "n_total": len(queries),
            "n_scored": len(scored_preds),
            "BLEU-4": bleu,
            "ROUGE-L": rouge,
            **bert,
            "sample_predictions": list(zip(queries[:10], predictions[:10], references[:10])),
        }
        out_path = f"results/oracle_{condition}_{lang}_results.json"
        json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        logger.info(f"{condition}/{lang}: BLEU-4={bleu:.4f} BERTScore-F={bert['BERTScore_F']:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
