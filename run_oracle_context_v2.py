"""
run_oracle_context_v2.py

Corrected oracle-context experiment (peer-review item 7).

v1 (run_oracle_context.py) fed a short synthetic "The answer is: X." sentence
as the oracle context. Diagnosis showed the mT5-small+LoRA generator ignores
very short/out-of-distribution contexts entirely (closed_book and oracle
produced byte-identical outputs for every seed), because all real
(training and retrieval) contexts are full encyclopedia-passage-length text.

v2 fixes this by using a genuine, realistic-length oracle context: the
actual corpus passage that contains the gold answer as a substring (the
same passage a perfect retriever could have returned), rather than a
synthetic one-line sentence. This is scored only on the subset of items
where such a passage exists (per results/answer_coverage_{lang}.json),
since no genuine oracle passage can be constructed for the rest.

Two conditions, both using the corrected mT5-small+LoRA generator with the
same sampling-decoding config as every other generation experiment:
  - "oracle_real": context = the real corpus passage containing the gold
    answer (top_k=1, same build_input format as the main pipeline).
  - "closed_book": no context (empty passage list), same item subset.
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


def gold_answer_strings(item, lang):
    ans_list = item.get("answers", {}).get(lang, [])
    out = []
    for a in ans_list:
        t = a.get("text")
        if t:
            out.append(t)
        for alias in a.get("aliases", []) or []:
            if alias:
                out.append(alias)
    return out


def find_oracle_passage(answers, corpus, lowered_texts):
    """Return the first corpus passage (dict) whose text contains any gold
    answer string as a substring (case-insensitive), or None."""
    for a in answers:
        al = a.lower()
        for p, lt in zip(corpus, lowered_texts):
            if al in lt:
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["ar", "ms"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    lang = args.lang
    corpus_file = f"indexes/corpus_{lang}.json"
    items = load_mkqa_first_n("indexes/mkqa.jsonl.gz", args.n)
    corpus = json.load(open(corpus_file, encoding="utf-8"))
    lowered_texts = [p["text"].lower() for p in corpus]

    end = args.end if args.end is not None else len(items)
    items = items[args.start:end]

    queries, references, oracle_passages = [], [], []
    for item in items:
        q = item["queries"].get(lang)
        answers = [a.strip() for a in gold_answer_strings(item, lang) if a and a.strip()]
        ref = answers[0] if answers else ""
        op = find_oracle_passage(answers, corpus, lowered_texts) if answers else None
        queries.append(q)
        references.append(ref)
        oracle_passages.append(op)

    n_covered = sum(1 for op in oracle_passages if op is not None)
    logger.info(f"[{lang}] slice [{args.start}:{end}]: {len(items)} items, {n_covered} with a genuine oracle passage.")

    gen = MT5Generator(use_lora=True)

    for condition in ["oracle_real", "closed_book"]:
        idxs = [i for i, op in enumerate(oracle_passages) if op is not None]
        preds, refs = [], []
        for i in idxs:
            q = queries[i]
            ref = references[i]
            if condition == "oracle_real":
                passages = [(oracle_passages[i], 1.0)]
            else:
                passages = []
            pred = gen.generate(q, passages, top_k=1)
            preds.append(pred)
            refs.append(ref)

        bleu = compute_bleu(preds, refs)
        rouge = compute_rouge_l(preds, refs)
        bert = compute_bertscore(preds, refs, lang=lang)

        out = {
            "condition": condition,
            "lang": lang,
            "slice": [args.start, end],
            "n_total_slice": len(items),
            "n_scored": len(preds),
            "BLEU-4": bleu,
            "ROUGE-L": rouge,
            **bert,
            "sample_predictions": list(zip([queries[i] for i in idxs[:10]], preds[:10], refs[:10])),
        }
        out_path = f"results/oracle2_{condition}_{lang}_{args.start}_{end}.json"
        json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        logger.info(f"{condition}/{lang}[{args.start}:{end}]: n={len(preds)} BLEU-4={bleu:.4f} BERTScore-F={bert['BERTScore_F']:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
