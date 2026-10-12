import json, sys
sys.path.insert(0, ".")
from evaluation.metrics import compute_bertscore, compute_bleu, compute_rouge_l
import random

def bootstrap_ci_mean(values, n_boot=2000, seed=42):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample)/n)
    means.sort()
    lo = means[int(0.025*n_boot)]
    hi = means[int(0.975*n_boot)]
    return lo, hi

for lang in ["ar","ms"]:
    for cond in ["oracle_real","closed_book"]:
        path = f"results/oracle2_gen_{cond}_{lang}.jsonl"
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        preds = [r["pred"] for r in rows]
        refs = [r["ref"] for r in rows]
        bleu = compute_bleu(preds, refs)
        rouge = compute_rouge_l(preds, refs)
        bert = compute_bertscore(preds, refs, lang=lang)
        out = {
            "condition": cond, "lang": lang, "n": len(rows),
            "BLEU-4": bleu, "ROUGE-L": rouge, **bert,
            "sample_predictions": [(r["query"], r["pred"], r["ref"]) for r in rows[:10]],
        }
        json.dump(out, open(f"results/oracle2_final_{cond}_{lang}.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"{lang}/{cond}: n={len(rows)} BLEU-4={bleu:.4f} ROUGE-L={rouge:.4f} BERTScore-F={bert['BERTScore_F']:.4f}")
