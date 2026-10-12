#!/usr/bin/env python3
"""Build blinded human-audit sheets for the answer-string relevance proxy (AR + MS).
Per language, 120 (question, passage) pairs: all string-matched mE5-large top-5 passages among the first 200 items (about 25), 60 oracle passages, 20 non-matched top-1 passages.
Annotators see question, gold answer, passage only (proxy status hidden)."""
import json, random, re, unicodedata, openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
R = "results/"
AR_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().translate(AR_DIG)
    s = re.sub(r"[ً-ٰٟـ]", "", s); s = re.sub("[أإآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    s = re.sub(r"(\d)\.0+\b", r"\1", s); s = re.sub(r"[^\w\s]", " ", s); return " ".join(s.split())
def match(p, ref):
    r = norm(ref); return bool(r) and re.search(r"(?<!\w)" + re.escape(r) + r"(?!\w)", norm(p)) is not None
rnd = random.Random(42); keys = []; sheets = {}
for L in ("ar", "ms"):
    d = json.load(open(R + f"table9_matched_retrieved_{L}.json")); S, RET = d["samples"], d["retrieved"]
    pas = json.load(open(R + f"oracle2_passages_{L}.json")); pre = json.load(open(R + f"oracle2_precomputed_{L}.json"))
    pre = pre["items"] if isinstance(pre["items"], list) else eval(pre["items"])
    pos, neg = [], []
    for i, (s, ret) in enumerate(zip(S, RET)):
        hit = [(k, r[0]["text"]) for k, r in enumerate(ret) if match(r[0]["text"], s["reference"])]
        if hit: pos.append((i, s, hit[0][0] + 1, hit[0][1]))
        elif ret: neg.append((i, s, 1, ret[0][0]["text"]))
    A = rnd.sample(pos, min(70, len(pos))); C = rnd.sample(neg, min(20, len(neg)))
    cand = [x for x in pre if str(x["i"]) in pas and x.get("oracle_idx") is not None]
    B = rnd.sample(cand, min(60, len(cand)))
    rows = [("A_string_matched_top5", s["query"], s["reference"], rank, t) for i, s, rank, t in A]
    rows += [("B_oracle_passage", x["query"], x["ref"], 0, pas[str(x["i"])]["text"]) for x in B]
    rows += [("C_not_matched_top1", s["query"], s["reference"], rank, t) for i, s, rank, t in C]
    rnd.shuffle(rows)
    for n, r in enumerate(rows, 1): keys.append(dict(lang=L, item=n, source=r[0], rank=r[3]))
    sheets[L] = rows
HDR = ["ID", "Question", "Gold answer (MKQA)", "Passage", "Judgment (pick from list)", "Notes (optional)"]
def build(path, rowsel):
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    ws0 = wb.create_sheet("Instructions")
    for t in open("audit/INSTRUCTIONS.txt", encoding="utf-8").read().split("\n"): ws0.append([t])
    ws0.column_dimensions["A"].width = 130
    for L, rows in sheets.items():
        ws = wb.create_sheet(L.upper()); ws.append(HDR)
        for c in ws[1]: c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="305496")
        dv = DataValidation(type="list", formula1='"SUPPORTS,MENTIONS_ONLY,NO"', allow_blank=True); ws.add_data_validation(dv)
        for n, r in enumerate(rows, 1):
            if rowsel is not None and n not in rowsel[L]: continue
            ws.append([f"{L}-{n:03d}", r[1], r[2], r[4], None, None]); dv.add(f"E{ws.max_row}")
        for col, w in zip("ABCDEF", (10, 40, 22, 110, 18, 30)): ws.column_dimensions[col].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row: c.alignment = Alignment(wrap_text=True, vertical="top", horizontal="right" if L == "ar" and c.column in (2, 3, 4) else "left")
        ws.freeze_panes = "A2"
    wb.save(path)
sub = {L: set(random.Random(7).sample(range(1, len(sheets[L]) + 1), 50)) for L in sheets}
build("audit/audit_rater1.xlsx", None); build("audit/audit_rater2.xlsx", sub)
json.dump(dict(keys=keys, rater2_subset={L: sorted(v) for L, v in sub.items()}), open("audit/audit_KEY_do_not_share.json", "w"), indent=1)
print({L: len(r) for L, r in sheets.items()}, {L: len(v) for L, v in sub.items()})
