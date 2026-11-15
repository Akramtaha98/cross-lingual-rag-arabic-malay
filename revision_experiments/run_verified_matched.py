#!/usr/bin/env python3
"""Matched verified-evidence comparison (answers Reviewer 1, round 2).
Same 29 questions (those with retrieved top-5), same model (Qwen2.5-7B-Instruct, HF bf16, greedy, batch size 1), same prompts.
Every window rule is GOLD-FREE: windows are taken from the start of a passage or chosen by overlap with the QUESTION, never with the answer.
Overall context budget = 2000 characters for both retrieved (5 passages x 400) and verified (1 passage x 2000), plus a tight 400-character variant.
Also records, for every question and condition, whether a gold answer/alias occurs in the text actually shown (evidence_in_context).
Usage (repo root): python3 revision_experiments/run_verified_matched.py [--ollama] [--dry]   (--ollama: local ollama server; --dry: no model)"""
import json, re, sys, time, unicodedata
DRY = "--dry" in sys.argv
OLLAMA = "--ollama" in sys.argv   # use the local ollama qwen2.5:7b (same pipeline as the stored pilot run); default is Hugging Face on CUDA
items = [x for x in json.load(open("results/verified_oracle_items.json", encoding="utf-8")) if x["top5"]]
assert len(items) == 29, len(items)
def prompt(L, q, ctx):
    if L == "ar":
        return (f"أجب عن السؤال باختصار (كلمات قليلة) اعتماداً على المقطع.\n\nالمقطع:\n{ctx}\n\nالسؤال: {q}\nالإجابة:" if ctx else f"أجب عن السؤال باختصار (كلمات قليلة).\n\nالسؤال: {q}\nالإجابة:")
    return (f"Jawab soalan dengan ringkas (beberapa perkataan) berdasarkan petikan.\n\nPetikan:\n{ctx}\n\nSoalan: {q}\nJawapan:" if ctx else f"Jawab soalan dengan ringkas (beberapa perkataan).\n\nSoalan: {q}\nJawapan:")
def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower()); t = re.sub(r"[ً-ْـ]", "", t); t = re.sub(r"[أإآ]", "ا", t)
    t = re.sub(r"(\d)\.0+\b", r"\1", t); t = re.sub(r"[^\w\s]", " ", t); return re.sub(r"\s+", " ", t).strip()
def has(text, golds):
    p = norm(text); return any(re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", p) for g in (norm(x) for x in golds) if g)
REF = re.compile(r"لا (?:يحتوي|يذكر|تحتوي|تذكر|توجد|يوجد)|غير مذكور|tidak (?:mengandungi|menyebut|menyatakan|dinyatakan|disebut)|tiada maklumat|not (?:mentioned|contain)")
def score(pred, golds):
    p = norm(pred); gs = [norm(g) for g in golds if norm(g)]
    return dict(em=any(p == g for g in gs), contains=any(re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", p) for g in gs), refusal=bool(REF.search(pred.lower())))
def start(text, w): return text[:w]
def qwin(text, q, w, step=25):
    """window of w characters with the largest number of distinct question words (length>=3); earliest on ties. Uses the question only."""
    if len(text) <= w: return text
    qs = {t for t in norm(q).split() if len(t) >= 3}
    best, bs = 0, -1
    for s in range(0, len(text) - w + 1, step):
        win = set(norm(text[s:s + w]).split()); sc = len(qs & win)
        if sc > bs: best, bs = s, sc
    return text[best:best + w]
def centered(text, golds, w=600):   # the OLD rule (uses the gold answer) - kept only for reference
    t = text.lower(); k = -1
    for g in golds:
        k = t.find(str(g).lower().strip())
        if k >= 0: break
    s = max(0, k - w) if k >= 0 else 0; return text[s:s + 2 * w]
def joined(ps): return "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(ps))
def contexts(x):
    q, vp, t5 = x["query"], x["verified_passage"], x["top5"]
    return {
      "closed_book": "",
      "retrieved_start_5x400": joined([start(t, 400) for t in t5]),
      "retrieved_qwin_5x400": joined([qwin(t, q, 400) for t in t5]),
      "verified_start_2000": start(vp, 2000),
      "verified_qwin_2000": qwin(vp, q, 2000),
      "verified_start_400": start(vp, 400),
      "verified_qwin_400": qwin(vp, q, 400),
      "verified_centered_1200_OLD_uses_gold": centered(vp, x["gold"]),
    }
jobs = []
for n, x in enumerate(items):
    for c, ctx in contexts(x).items(): jobs.append((n, x, c, ctx))
if not DRY and not OLLAMA:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name = "Qwen/Qwen2.5-7B-Instruct"; tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
rows = []; t0 = time.time()
for k, (n, x, c, ctx) in enumerate(jobs):
    pred = ""
    if OLLAMA and not DRY:
        sys.path.insert(0, "."); from run_strong_generator import generate_ollama
        pred = generate_ollama(prompt(x["lang"], x["query"], ctx), max_tokens=60).strip()
        if k % 20 == 0: print(k, "/", len(jobs), round(time.time() - t0), "s", flush=True)
    elif not DRY:
        txt = tok.apply_chat_template([{"role": "user", "content": prompt(x["lang"], x["query"], ctx)}], tokenize=False, add_generation_prompt=True)
        enc = tok([txt], return_tensors="pt").to("cuda:0")
        with torch.no_grad(): g = model.generate(**enc, max_new_tokens=60, do_sample=False)
        pred = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if k % 20 == 0: print(k, "/", len(jobs), round(time.time() - t0), "s", flush=True)
    vp_shown = (c.startswith("retrieved") and any(t[:100] == x["verified_passage"][:100] for t in x["top5"]))
    rows.append(dict(n=n, lang=x["lang"], cond=c, query=x["query"], gold=x["gold"][:5], pred=pred, context=ctx, shown_chars=len(ctx),
                     evidence_in_context=has(ctx, x["gold"]) if ctx else False, verified_passage_among_top5=vp_shown, **(score(pred, x["gold"]) if not DRY else {})))
out = ("results/verified_matched_out_ollama.json" if OLLAMA else "results/verified_matched_out.json") if not DRY else "/tmp/verified_matched_dry.json"
json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
summ = {}
for scope, sel in (("AR", lambda r: r["lang"] == "ar"), ("MS", lambda r: r["lang"] == "ms"), ("matched_29", lambda r: True)):
    for c in contexts(items[0]):
        r = [x for x in rows if x["cond"] == c and sel(x)]
        summ[f"{scope}|{c}"] = dict(n=len(r), contains=sum(x.get("contains", 0) for x in r), em=sum(x.get("em", 0) for x in r), refusals=sum(x.get("refusal", 0) for x in r),
                                     evidence_in_context=sum(x["evidence_in_context"] for x in r), mean_chars=round(sum(x["shown_chars"] for x in r) / max(1, len(r))))
json.dump(summ, open(out.replace("_out", "_summary").replace("/tmp/verified_matched_dry", "/tmp/verified_matched_dry_summary"), "w"), indent=1)
for k, v in summ.items():
    if k.startswith("matched_29"): print(k, v)
print("DONE", "(dry run: no model)" if DRY else "")
