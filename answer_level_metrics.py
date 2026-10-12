#!/usr/bin/env python3
"""Answer-level correctness: normalized Exact Match, token F1, and gold-contained-in-output rate."""
import json, re, unicodedata, collections, sys
AR_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().translate(AR_DIG)
    s = re.sub(r"[ً-ٰٟـ]", "", s)              # Arabic diacritics, tatweel
    s = re.sub("[أإآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    s = re.sub(r"(\d)\.0+\b", r"\1", s)                            # 11.0 -> 11
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())
def f1(p, r):
    p, r = norm(p).split(), norm(r).split()
    if not p or not r: return float(p == r)
    c = sum((collections.Counter(p) & collections.Counter(r)).values())
    if c == 0: return 0.0
    pr, rc = c/len(p), c/len(r); return 2*pr*rc/(pr+rc)
def em(p, r): return float(norm(p) == norm(r))
def has(p, r):
    r = norm(r); return float(bool(r) and re.search(r"(?<!\w)"+re.escape(r)+r"(?!\w)", norm(p)) is not None)
def score(pairs):
    n = len(pairs)
    return dict(n=n, EM=sum(em(p,r) for p,r in pairs)/n, F1=sum(f1(p,r) for p,r in pairs)/n, Contains=sum(has(p,r) for p,r in pairs)/n)
if __name__ == "__main__":
    out = {}
    R = "results/"
    for lang in ("ar","ms"):
        for name, f in [("mt5_base_matched", f"table9_matched_me5large_mt5_base_{lang}_results.json"),
                        ("mt5_lora_matched", f"table9_matched_me5large_mt5_lora_{lang}_results.json"),
                        ("qwen_matched", f"strong_gen_qwen2.5_7b_{lang}_results.json")]:
            d = json.load(open(R+f)); out[f"{name}_{lang}"] = score([(s["answer"], s["reference"]) for s in d["samples"]])
        for cond in ("closed_book","oracle_real"):
            rows = [json.loads(l) for l in open(R+f"oracle2_gen_{cond}_{lang}.jsonl")]
            out[f"{cond}_{lang}"] = score([(x["pred"], x["ref"]) for x in rows])
    json.dump(out, open(R+"answer_level_metrics.json","w"), indent=2)
    for k,v in out.items(): print(k, {a:(round(b,3) if isinstance(b,float) else b) for a,b in v.items()})
