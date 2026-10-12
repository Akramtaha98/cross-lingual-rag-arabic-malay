"""
Recompute ROUGE-L for all EXP1 result files using a script-agnostic tokenizer
(plain Unicode whitespace split) instead of rouge_score's DefaultTokenizer,
which strips all non-[a-z0-9] characters and silently zeroes out ROUGE-L
for Arabic (and any other non-Latin-script) text.
"""
import json, glob, os, sys
from rouge_score import rouge_scorer, tokenizers

class UnicodeWhitespaceTokenizer(tokenizers.Tokenizer):
    def tokenize(self, text):
        return text.lower().split()

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=UnicodeWhitespaceTokenizer())

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
files = sorted(glob.glob(os.path.join(RESULTS_DIR, "exp1_*_results.json")))

out = {}
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    samples = d.get("samples", [])
    scores = []
    for s in samples:
        pred = s.get("answer", "") or ""
        ref = s.get("reference", "") or ""
        if not pred.strip() or not ref.strip():
            scores.append(0.0)
            continue
        scores.append(scorer.score(ref, pred)["rougeL"].fmeasure)
    mean = sum(scores) / len(scores) if scores else 0.0
    old = d["metrics"].get("ROUGE-L")
    out[os.path.basename(f)] = {"n": len(scores), "old_rouge_l": old, "new_rouge_l": round(mean, 6)}
    print(f"{os.path.basename(f):40s} n={len(scores):5d}  old={old!s:>10}  new={mean:.6f}")

json.dump(out, open(os.path.join(RESULTS_DIR, "rouge_l_recomputed.json"), "w"), indent=2)
