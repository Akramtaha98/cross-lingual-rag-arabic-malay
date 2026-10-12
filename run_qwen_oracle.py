#!/usr/bin/env python3
"""Qwen2.5-7B-Instruct (Ollama) under closed-book vs. oracle-context vs. mE5-large-retrieved context, on the
same answer-coverage-positive items. The oracle context is a window around the gold-answer occurrence in the
real corpus passage, so the answer is guaranteed visible. Resumable (jsonl checkpoint).
usage: python3 run_qwen_oracle.py --lang ar --n 100
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, ".")
from run_strong_generator import generate_ollama
from answer_level_metrics import norm, score
ap = argparse.ArgumentParser(); ap.add_argument("--lang", required=True); ap.add_argument("--n", type=int, default=100)
a = ap.parse_args(); L = a.lang
pre = json.load(open(f"results/oracle2_precomputed_{L}.json"))["items"]
pas = json.load(open(f"results/oracle2_passages_{L}.json"))
items = [x for x in pre if x.get("oracle_idx") not in (None, "", "None")][: a.n]
def window(text, ref, w=600):
    t = text.lower(); r = str(ref).lower().strip(); k = t.find(r)
    if k < 0: k = max(0, t.find(re.sub(r"\.0+\b", "", r)))
    s = max(0, k - w); return text[s:s + 2 * w]
def prompt(q, ctx):
    if L == "ar":
        if ctx: return f"أجب عن السؤال باختصار (كلمات قليلة) اعتماداً على المقطع.\n\nالمقطع:\n{ctx}\n\nالسؤال: {q}\nالإجابة:"
        return f"أجب عن السؤال باختصار (كلمات قليلة).\n\nالسؤال: {q}\nالإجابة:"
    if ctx: return f"Jawab soalan dengan ringkas (beberapa perkataan) berdasarkan petikan.\n\nPetikan:\n{ctx}\n\nSoalan: {q}\nJawapan:"
    return f"Jawab soalan dengan ringkas (beberapa perkataan).\n\nSoalan: {q}\nJawapan:"
ck = f"results/_qwen_oracle_{L}.jsonl"
done = {(d["i"], d["cond"]): d for d in map(json.loads, open(ck))} if os.path.exists(ck) else {}
f = open(ck, "a", encoding="utf-8"); t0 = time.time()
for x in items:
    for cond in ("closed_book", "oracle"):
        if (x["i"], cond) in done: continue
        ctx = window(pas[str(x["i"])]["text"], x["ref"]) if cond == "oracle" else ""
        pred = generate_ollama(prompt(x["query"], ctx), max_tokens=60)
        d = {"i": x["i"], "cond": cond, "query": x["query"], "ref": x["ref"], "pred": pred}
        f.write(json.dumps(d, ensure_ascii=False) + "\n"); f.flush(); done[(x["i"], cond)] = d
    print(len(done) // 2, "items", round(time.time() - t0), "s", flush=True)
out = {c: score([(d["pred"], d["ref"]) for (i, cc), d in done.items() if cc == c]) for c in ("closed_book", "oracle")}
json.dump(out, open(f"results/qwen_oracle_{L}.json", "w"), indent=2); print("DONE", L, out)
