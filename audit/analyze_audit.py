#!/usr/bin/env python3
"""Usage: python3 audit/analyze_audit.py audit_rater1_done.xlsx [audit_rater2_done.xlsx]
Reports proxy precision (share of string-matched passages labeled SUPPORTS), the false-positive rate with Wilson 95% CIs,
the SUPPORTS rate among non-matched passages (rough false-negative indication), and Cohen's kappa between raters."""
import json, sys, math, collections, openpyxl
KEY = json.load(open("audit/audit_KEY_do_not_share.json"))["keys"]
src = {(k["lang"], k["item"]): k["source"] for k in KEY}
def load(p):
    out = {}; wb = openpyxl.load_workbook(p)
    for L in ("ar", "ms"):
        for row in wb[L.upper()].iter_rows(min_row=2, values_only=True):
            if row[0] and row[4]: out[(L, int(row[0].split("-")[1]))] = row[4].strip().upper()
    return out
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
r1 = load(sys.argv[1])
for L in ("ar", "ms"):
    for name, grp in (("string-matched (A+B)", ("A", "B")), ("non-matched (C)", ("C",))):
        ids = [k for k in r1 if k[0] == L and src[k][0] in grp]
        s = sum(r1[k] == "SUPPORTS" for k in ids); lo, hi = wilson(s, len(ids))
        print(f"{L.upper()} {name}: n={len(ids)} SUPPORTS={s} ({s/max(len(ids),1):.1%}, 95% CI {lo:.1%}-{hi:.1%}); false-positive rate among matched = {1 - s/max(len(ids),1):.1%}" if grp[0] != "C" else f"{L.upper()} {name}: n={len(ids)} SUPPORTS={s} ({s/max(len(ids),1):.1%}, 95% CI {lo:.1%}-{hi:.1%})")
    for g in ("A", "B"):
        ids = [k for k in r1 if k[0] == L and src[k].startswith(g)]; s = sum(r1[k] == "SUPPORTS" for k in ids)
        print(f"   source {g}: n={len(ids)} SUPPORTS={s}")
if len(sys.argv) > 2:
    r2 = load(sys.argv[2]); com = [k for k in r2 if k in r1]
    po = sum(r1[k] == r2[k] for k in com) / len(com); c1 = collections.Counter(r1[k] for k in com); c2 = collections.Counter(r2[k] for k in com)
    pe = sum(c1[x] * c2[x] for x in set(c1) | set(c2)) / len(com) ** 2
    print(f"Agreement on {len(com)} shared items: {po:.1%}; Cohen's kappa = {(po - pe) / (1 - pe):.2f}")
