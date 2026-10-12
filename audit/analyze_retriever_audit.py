#!/usr/bin/env python3
"""Usage: python3 audit/analyze_retriever_audit.py audit/retriever_audit_rater1_done.xlsx
Human-labeled Hit@3 (>=1 SUPPORTS passage in top-3) and P@3 per retriever, Wilson CIs, paired bootstrap (question level)."""
import json, sys, math, random, openpyxl
KEY = json.load(open("audit/retriever_audit_KEY_do_not_share.json"))
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d
wb = openpyxl.load_workbook(sys.argv[1]); R = ("bm25", "labse", "me5large")
for L in ("ar", "ms"):
    lab = {int(r[0].split("-")[1]): str(r[4]).strip().upper() for r in wb[L.upper()].iter_rows(min_row=2, values_only=True) if r[0] and r[4]}
    qs = sorted({k["q"] for k in KEY[L]}); hit = {r: {q: 0 for q in qs} for r in R}; sup = {r: {q: 0 for q in qs} for r in R}; tot = {r: 0 for r in R}
    for k in KEY[L]:
        for r, rank in k["ranks"].items():
            tot[r] += 1; s = lab.get(k["item"]) == "SUPPORTS"; sup[r][k["q"]] += s; hit[r][k["q"]] |= s
    print(f"== {L.upper()} ({len(qs)} questions, {len(lab)} labeled pairs)")
    for r in R:
        h = sum(hit[r].values()); p = sum(sup[r].values()); lo, hi = wilson(h, len(qs)); plo, phi = wilson(p, tot[r])
        print(f"  {r:9s} Hit@3 {h}/{len(qs)} = {h/len(qs):.1%} [{lo:.1%}-{hi:.1%}]  P@3 {p}/{tot[r]} = {p/tot[r]:.1%} [{plo:.1%}-{phi:.1%}]")
    rnd = random.Random(42)
    for a, b in (("me5large", "bm25"), ("me5large", "labse"), ("bm25", "labse")):
        d = [hit[a][q] - hit[b][q] for q in qs]; bs = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(5000))
        print(f"  Hit@3 {a}-{b}: {sum(d)/len(d):+.3f} [{bs[125]:+.3f},{bs[4874]:+.3f}]")
