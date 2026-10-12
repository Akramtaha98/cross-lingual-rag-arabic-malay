import json, random
from bert_score import score as bert_score_fn

def load(lang, cond):
    rows = [json.loads(l) for l in open(f"results/oracle2_gen_{cond}_{lang}.jsonl", encoding="utf-8")]
    return {r["i"]: r for r in rows}

results = {}
for lang in ["ar","ms"]:
    o = load(lang, "oracle_real")
    c = load(lang, "closed_book")
    common = sorted(set(o.keys()) & set(c.keys()))
    preds_o = [o[i]["pred"] for i in common]
    preds_c = [c[i]["pred"] for i in common]
    refs = [o[i]["ref"] for i in common]

    Po, Ro, Fo = bert_score_fn(preds_o, refs, model_type="bert-base-multilingual-cased", lang=lang, verbose=False, rescale_with_baseline=False)
    Pc, Rc, Fc = bert_score_fn(preds_c, refs, model_type="bert-base-multilingual-cased", lang=lang, verbose=False, rescale_with_baseline=False)
    Fo = Fo.tolist(); Fc = Fc.tolist()
    diffs = [a-b for a,b in zip(Fo, Fc)]
    n = len(diffs)
    mean_diff = sum(diffs)/n

    rng = random.Random(42)
    boot_means = []
    for _ in range(2000):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample)/n)
    boot_means.sort()
    lo = boot_means[int(0.025*2000)]
    hi = boot_means[int(0.975*2000)]
    p_ge_0 = sum(1 for b in boot_means if b <= 0)/2000  # fraction of resamples where diff <=0 (oracle not better)
    results[lang] = {"n": n, "mean_F_oracle": sum(Fo)/n, "mean_F_closed": sum(Fc)/n,
                      "mean_diff_oracle_minus_closed": mean_diff, "ci95": [lo, hi],
                      "frac_boot_le_0": p_ge_0}
    print(lang, results[lang])

json.dump(results, open("results/oracle2_paired_significance.json","w"), indent=2)
