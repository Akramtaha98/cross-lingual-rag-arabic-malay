#!/usr/bin/env python3
"""Builds results/verified_oracle_items.json: questions with a human-verified SUPPORTS passage (Rater 1 labels, with adjudicated
overrides on the shared pairs), from both audits. Adds MKQA aliases and, where available, mE5-large top-5 retrieval."""
import gzip, json, re, openpyxl, collections
def load(p, col=4):
    out = {}; wb = openpyxl.load_workbook(p)
    for L in ("ar", "ms"):
        for r in wb[L.upper()].iter_rows(min_row=2, values_only=True):
            if r[0] and r[col]: out[(L, int(r[0].split("-")[1]))] = (r[1], r[2], r[3], str(r[col]).strip().upper())
    return out
mk = {}
for line in gzip.open("indexes/mkqa.jsonl.gz", "rt", encoding="utf-8"):
    d = json.loads(line); mk[d["query"] if isinstance(d["query"], str) else str(d["query"])] = d
def gold(d, L):
    return [x for a in d["answers"][L] for x in [a["text"]] + list(a.get("aliases") or [])]
adj = {k: v[3] for k, v in load("audit/adjudication_sheet_completed.xlsx", 6).items()}
ver = collections.defaultdict(dict)   # (lang, query) -> {...}
# audit 1: Rater 1 sheet (A retrieved matches, B oracle)
for k, (q, g, p, lab) in load("audit/audit_rater1_done.xlsx").items():
    lab = adj.get(k, lab)
    if lab == "SUPPORTS" and p and not str(p).startswith("="): ver[(k[0], q)].setdefault("passages", []).append(p)
# audit 2: retriever audit Rater 1
for k, (q, g, p, lab) in load("audit/retriever_audit_rater1_done.xlsx").items():
    if lab == "SUPPORTS": ver[(k[0], q)].setdefault("passages", []).append(p)
items = []
ret = {L: json.load(open(f"results/table9_matched_retrieved_{L}.json", encoding="utf-8")) for L in ("ar", "ms")}
for (L, q), v in sorted(ver.items()):
    # query text in sheets is the MKQA query in language L; find the record
    rec = next((r for r in ({} if False else [])), None)
    items.append((L, q, v["passages"]))
# map queries to MKQA records by language-specific query text
byq = {L: {} for L in ("ar", "ms")}
for line in gzip.open("indexes/mkqa.jsonl.gz", "rt", encoding="utf-8"):
    d = json.loads(line); byq["ar"].setdefault(d["queries"]["ar"], d) if "queries" in d else None; byq["ms"].setdefault(d["queries"]["ms"], d) if "queries" in d else None
out = []
for L, q, ps in items:
    d = byq[L].get(q)
    if d is None: print("no MKQA match:", L, q[:50]); continue
    S = ret[L]["samples"]; idx = next((i for i, s in enumerate(S) if s["query"] == q), None)
    out.append(dict(lang=L, query=q, gold=gold(d, L), verified_passage=ps[0], n_verified=len(ps), top5=[r[0]["text"] if isinstance(r[0], dict) else r[0] for r in ret[L]["retrieved"][idx]] if idx is not None else None))
json.dump(out, open("results/verified_oracle_items.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print({L: sum(o["lang"] == L for o in out) for L in ("ar", "ms")}, "with top5:", {L: sum(o["lang"] == L and o["top5"] is not None for o in out) for L in ("ar", "ms")})
