#!/usr/bin/env python3
"""Adjudicated audit statistics. Usage: python3 audit/analyze_adjudicated.py
Merges Rater 1 labels, Rater 2 labels and adjudication (audit/adjudication_sheet_completed.xlsx).
Adjudicated label = agreed label on shared pairs (agreements kept, disputes replaced by the adjudicated label)."""
import json, math, collections, openpyxl
KEY = json.load(open("audit/audit_KEY_do_not_share.json"))["keys"]; src = {(k["lang"], k["item"]): k["source"][0] for k in KEY}
def load(p, col=4):
    out = {}; wb = openpyxl.load_workbook(p)
    for L in ("ar", "ms"):
        for r in wb[L.upper()].iter_rows(min_row=2, values_only=True):
            if r[0] and r[col]: out[(L, int(r[0].split("-")[1]))] = r[col].strip().upper()
    return out
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d
r1 = load("audit/audit_rater1_done.xlsx"); r2 = load("audit/audit_rater2_done.xlsx"); adj = load("audit/adjudication_sheet_completed.xlsx", 6)
shared = [k for k in r2 if k in r1 and k != ("ms", 42)]
final = {k: (adj[k] if k in adj else r1[k]) for k in shared}
print("shared:", len(shared), "disputed adjudicated:", len(adj), {L: sum(k[0] == L for k in shared) for L in ("ar", "ms")})
print("adjudicated label counts on shared:", collections.Counter(final.values()))
for L in ("ar", "ms"):
    for g, name in (("AB", "string-matched (A+B)"), ("A", "retrieved A"), ("B", "oracle B")):
        ids = [k for k in shared if k[0] == L and src[k] in g]
        if not ids: continue
        row = []
        for nm, lab in (("R1", r1), ("R2", r2), ("adjudicated", final)):
            s = sum(lab[k] == "SUPPORTS" for k in ids); lo, hi = wilson(s, len(ids)); row.append(f"{nm} {s}/{len(ids)} ({s/len(ids):.1%}; {lo:.0%}-{hi:.0%})")
        print(L.upper(), name, " | ".join(row))
# combined adjudicated sample over all Rater 1 items (adjudicated labels override where available)
full = {k: (final[k] if k in final else r1[k]) for k in r1 if k != ("ms", 42)}
print("\nRater-1 sample with adjudicated labels substituted on the shared pairs (labels from adjudication where shared; else Rater 1):")
for L in ("ar", "ms"):
    for g, name in (("AB", "string-matched pooled"), ("A", "retrieved A"), ("B", "oracle B")):
        ids = [k for k in full if k[0] == L and src[k] in g]; s = sum(full[k] == "SUPPORTS" for k in ids); lo, hi = wilson(s, len(ids))
        s1 = sum(r1[k] == "SUPPORTS" for k in ids)
        print(f"{L.upper()} {name}: R1-only {s1}/{len(ids)} ({s1/len(ids):.1%}) -> adjudicated-where-available {s}/{len(ids)} ({s/len(ids):.1%}; {lo:.0%}-{hi:.0%})")
    ids = [k for k in full if k[0] == L and src[k] == "A"]
    print(f"{L.upper()} supported-Hit@5 lower bound (adjudicated where available): {sum(full[k]=='SUPPORTS' for k in ids)}/200")
# agreement with adjudicated labels
for nm, lab in (("R1", r1), ("R2", r2)):
    print(nm, "agrees with adjudicated label on", sum(lab[k] == final[k] for k in shared), "/", len(shared))
