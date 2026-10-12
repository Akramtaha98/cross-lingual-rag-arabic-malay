#!/usr/bin/env python3
"""Verified-evidence pilot. Same Qwen2.5-7B-Instruct, same prompt template, greedy decoding, three conditions per question:
closed_book (no context), retrieved (mE5-large top-5, 400 chars each), verified (human-confirmed SUPPORTS passage).
Scoring uses all MKQA aliases. Usage: python3 run_verified_oracle_hf.py   (needs results/verified_oracle_items.json)"""
import json, re, sys, time, unicodedata, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
items = json.load(open("results/verified_oracle_items.json", encoding="utf-8"))
def prompt(L, q, ctx):
    if L == "ar":
        return (f"أجب عن السؤال باختصار (كلمات قليلة) اعتماداً على المقطع.\n\nالمقطع:\n{ctx}\n\nالسؤال: {q}\nالإجابة:" if ctx else f"أجب عن السؤال باختصار (كلمات قليلة).\n\nالسؤال: {q}\nالإجابة:")
    return (f"Jawab soalan dengan ringkas (beberapa perkataan) berdasarkan petikan.\n\nPetikan:\n{ctx}\n\nSoalan: {q}\nJawapan:" if ctx else f"Jawab soalan dengan ringkas (beberapa perkataan).\n\nSoalan: {q}\nJawapan:")
def window(text, golds, w=600):
    t = text.lower(); k = -1
    for g in golds:
        k = t.find(str(g).lower().strip())
        if k >= 0: break
    s = max(0, k - w) if k >= 0 else 0; return text[s:s + 2 * w]
def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower()); t = re.sub(r"[ً-ْـ]", "", t); t = re.sub(r"[أإآ]", "ا", t)
    t = re.sub(r"(\d)\.0+\b", r"\1", t); t = re.sub(r"[^\w\s]", " ", t); return re.sub(r"\s+", " ", t).strip()
REF = re.compile(r"لا (?:يحتوي|يذكر|تحتوي|تذكر|توجد|يوجد)|غير مذكور|tidak (?:mengandungi|menyebut|menyatakan|dinyatakan|disebut)|tiada maklumat|not (?:mentioned|contain)")
def score(pred, golds):
    p = norm(pred); gs = [norm(g) for g in golds if norm(g)]
    return dict(em=any(p == g for g in gs), contains=any(re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", p) for g in gs), refusal=bool(REF.search(pred.lower())))
name = "Qwen/Qwen2.5-7B-Instruct"; tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
jobs = []
for n, x in enumerate(items):
    ctxs = {"closed_book": ""}
    if x["top5"]: ctxs["retrieved"] = "\n\n".join(f"[{i+1}] {t[:400]}" for i, t in enumerate(x["top5"]))
    ctxs["verified"] = window(x["verified_passage"], x["gold"])
    for c, ctx in ctxs.items():
        jobs.append((n, x, c, tok.apply_chat_template([{"role": "user", "content": prompt(x["lang"], x["query"], ctx)}], tokenize=False, add_generation_prompt=True)))
rows = []; B = 8; t0 = time.time()
for i in range(0, len(jobs), B):
    ch = jobs[i:i + B]; enc = tok([c[3] for c in ch], return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad(): g = model.generate(**enc, max_new_tokens=60, do_sample=False)
    for c, o in zip(ch, g):
        pred = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        rows.append(dict(n=c[0], lang=c[1]["lang"], cond=c[2], query=c[1]["query"], gold=c[1]["gold"][:3], pred=pred, **score(pred, c[1]["gold"])))
    print(len(rows), "/", len(jobs), round(time.time() - t0), "s", flush=True)
json.dump(rows, open("results/verified_oracle_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for L in ("ar", "ms"):
    for c in ("closed_book", "retrieved", "verified"):
        r = [x for x in rows if x["lang"] == L and x["cond"] == c]
        if r: print(L, c, f"n={len(r)} contains={sum(x['contains'] for x in r)} EM={sum(x['em'] for x in r)} refusals={sum(x['refusal'] for x in r)}")
print("DONE")
