#!/usr/bin/env python3
"""Builds audit_for_raters/adjudication_sheet.xlsx: every shared item where Rater 1 and Rater 2 disagreed.
Labels are shown as 'Label X / Label Y' in random order (rater identity hidden). Raters meet, discuss, and record ONE agreed label."""
import json, random, openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
def load(p):
    out = {}; wb = openpyxl.load_workbook(p)
    for L in ("ar", "ms"):
        for row in wb[L.upper()].iter_rows(min_row=2, values_only=True):
            if row[0] and row[4]: out[(L, int(row[0].split("-")[1]))] = (row[1], row[2], row[3], row[4].strip().upper())
    return out
r1 = load("audit/audit_rater1_done.xlsx"); r2 = load("audit/audit_rater2_done.xlsx")
rnd = random.Random(7)
dis = [k for k in sorted(r2) if k in r1 and k != ("ms", 42) and r1[k][3] != r2[k][3] and r1[k][2] and not str(r1[k][2]).startswith("=")]
print("disputed:", len(dis), {L: sum(k[0] == L for k in dis) for L in ("ar", "ms")})
RUBRIC = """ADJUDICATION INSTRUCTIONS (both annotators together)
Goal: agree ONE final label for each item below where you originally disagreed. Your original independent labels are kept separately for the agreement statistics; do not change them.
Use this tightened rule:
SUPPORTS      = Reading ONLY the passage, a reader could answer the question and obtain the gold answer (or an equivalent form of it) as the answer to THIS question. The passage must state the fact about the entity/event the question asks about. A passage that only lists the same number/year/word in an unrelated context is NOT SUPPORTS.
MENTIONS_ONLY = The passage contains the gold answer text (or a form of it), but about something else, or only loosely related so that the question cannot be answered from it.
NO            = The passage does not contain the gold answer and does not answer the question.
Tie-break test: 'If I had only this passage, could I confidently answer the question?' Yes -> SUPPORTS; no, but the answer string appears -> MENTIONS_ONLY; neither -> NO.
Fill 'Final agreed label' and write one short reason. Do not edit other columns."""
wb = openpyxl.Workbook(); wb.remove(wb.active); ws0 = wb.create_sheet("Instructions")
for t in RUBRIC.split("\n"): ws0.append([t])
ws0.column_dimensions["A"].width = 140
for L in ("ar", "ms"):
    ws = wb.create_sheet(L.upper()); ws.append(["ID", "Question", "Gold answer (MKQA)", "Passage", "Label X", "Label Y", "Final agreed label", "Reason (short)"])
    for c in ws[1]: c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="305496")
    dv = DataValidation(type="list", formula1='"SUPPORTS,MENTIONS_ONLY,NO"', allow_blank=True); ws.add_data_validation(dv)
    for k in [k for k in dis if k[0] == L]:
        a, b = r1[k][3], r2[k][3]; x, y = (a, b) if rnd.random() < .5 else (b, a)
        ws.append([f"{L}-{k[1]:03d}", r1[k][0], r1[k][1], r1[k][2], x, y, None, None]); dv.add(ws.cell(ws.max_row, 7))
    for col, w in zip("ABCDEFGH", (10, 45, 25, 100, 16, 16, 20, 40)): ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row: c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
wb.save("../audit_for_raters/adjudication_sheet.xlsx")
json.dump([list(k) for k in dis], open("audit/adjudication_items_do_not_share.json", "w"))
