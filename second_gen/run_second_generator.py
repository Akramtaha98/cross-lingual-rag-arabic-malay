#!/usr/bin/env python3
"""Second-generator study on one H200.
For each model: (A) 200 matched questions per language with the same retrieved context and prompt as the Qwen row of
Table 9 (BERTScore-F with bootstrap CI, EM, Contains-gold, refusals), paired against the stored Qwen2.5-7B (ollama) and
mT5+LoRA answers; (B) the 37-question verified-evidence pilot (closed-book / retrieved / verified), same prompts as
run_verified_oracle_hf.py.
Usage:  python run_second_generator.py [model ...]
Default models: Qwen2.5-7B-Instruct (control, same HF pipeline), Qwen2.5-3B-Instruct, Qwen2.5-14B-Instruct.
Input:  second_gen_inputs.json.gz (in this folder).  Run from this folder.  Output: results_second_gen/<model>.json"""
import gzip, json, os, re, sys, time, unicodedata
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = sys.argv[1:] or ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]
DATA = json.load(gzip.open("second_gen_inputs.json.gz", "rt", encoding="utf-8"))
os.makedirs("results_second_gen", exist_ok=True)
BATCH, MAXNEW = 16, 120


def rag_prompt(q, ctx, lang):  # same as run_strong_generator.py
    if lang == "ar":
        return ("أنت مساعد مفيد يجيب على الأسئلة باللغة العربية.\nاستخدم فقط المقاطع المقدمة للإجابة. إذا لم تكن الإجابة في المقاطع، قل «لا أعلم».\n\n"
                f"المقاطع:\n{ctx}\n\nالسؤال: {q}\nالإجابة:")
    return ("Anda adalah pembantu yang berguna yang menjawab soalan dalam Bahasa Melayu.\nGunakan hanya petikan yang diberikan untuk menjawab. Jika jawapan tiada dalam petikan, katakan «Tidak tahu».\n\n"
            f"Petikan:\n{ctx}\n\nSoalan: {q}\nJawapan:")


def pilot_prompt(L, q, ctx):  # same as run_verified_oracle_hf.py
    if L == "ar":
        return (f"أجب عن السؤال باختصار (كلمات قليلة) اعتماداً على المقطع.\n\nالمقطع:\n{ctx}\n\nالسؤال: {q}\nالإجابة:" if ctx else f"أجب عن السؤال باختصار (كلمات قليلة).\n\nالسؤال: {q}\nالإجابة:")
    return (f"Jawab soalan dengan ringkas (beberapa perkataan) berdasarkan petikan.\n\nPetikan:\n{ctx}\n\nSoalan: {q}\nJawapan:" if ctx else f"Jawab soalan dengan ringkas (beberapa perkataan).\n\nSoalan: {q}\nJawapan:")


def window(text, golds, w=600):
    t = text.lower(); k = -1
    for g in golds:
        k = t.find(str(g).lower().strip())
        if k >= 0: break
    s = max(0, k - w) if k >= 0 else 0
    return text[s:s + 2 * w]


def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower()); t = re.sub(r"[ً-ْـ]", "", t); t = re.sub(r"[أإآ]", "ا", t)
    t = re.sub(r"(\d)\.0+\b", r"\1", t); t = re.sub(r"[^\w\s]", " ", t); return re.sub(r"\s+", " ", t).strip()


REF = re.compile(r"لا (?:يحتوي|يذكر|تحتوي|تذكر|توجد|يوجد|أعلم|اعلم)|غير مذكور|tidak (?:mengandungi|menyebut|menyatakan|dinyatakan|disebut|tahu)|tiada maklumat|not (?:mentioned|contain)")


def score(pred, golds):
    p = norm(pred); gs = [norm(g) for g in golds if norm(g)]
    return dict(em=any(p == g for g in gs), contains=any(re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", p) for g in gs),
                refusal=bool(REF.search(pred.lower())))


def generate(tok, model, prompts):
    outs = []
    for i in range(0, len(prompts), BATCH):
        chunk = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts[i:i + BATCH]]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in g]
        print(f"  {len(outs)}/{len(prompts)}", flush=True)
    return outs


def bertscore(preds, refs, lang):
    from bert_score import score as bs
    keep = [i for i, p in enumerate(preds) if str(p).strip()]
    out = np.zeros(len(preds), dtype=np.float32)  # an empty output scores 0
    if keep:
        _, _, F = bs([preds[i] for i in keep], [refs[i] for i in keep], model_type="bert-base-multilingual-cased", lang=lang, verbose=False)
        out[keep] = F.numpy()
    return out


def ci(x, rng, B=2000):
    x = np.asarray(x); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    rng = np.random.default_rng(0)
    for name in MODELS:
        short = name.split("/")[-1]; t0 = time.time(); print("=== ", name, flush=True)
        tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda:0")
        res = {"model": name, "matched": {}, "pilot": {}}
        for L, rows in DATA["matched"].items():
            preds = generate(tok, model, [rag_prompt(r["query"], r["context"], L) for r in rows])
            refs = [r["reference"] for r in rows]
            F = bertscore(preds, refs, L); F7 = bertscore([r["qwen7b_ollama"] for r in rows], refs, L); Fm = bertscore([r["mt5_lora"] for r in rows], refs, L)
            sc = [score(p, [r]) for p, r in zip(preds, refs)]
            d7, dm = F - F7, F - Fm
            res["matched"][L] = dict(n=len(rows), BERTScore_F=float(F.mean()), ci=ci(F, rng),
                                     minus_qwen7b_ollama=dict(mean=float(d7.mean()), ci=ci(d7, rng)),
                                     minus_mt5_lora=dict(mean=float(dm.mean()), ci=ci(dm, rng)),
                                     EM=sum(s["em"] for s in sc), contains=sum(s["contains"] for s in sc), refusals=sum(s["refusal"] for s in sc),
                                     preds=preds)
            m = res["matched"][L]
            print(f"[{L}] BERTScore-F {m['BERTScore_F']:.3f} {m['ci']}  vs Qwen7B-ollama {m['minus_qwen7b_ollama']['mean']:+.3f} {m['minus_qwen7b_ollama']['ci']}  vs mT5+LoRA {m['minus_mt5_lora']['mean']:+.3f} {m['minus_mt5_lora']['ci']}  EM {m['EM']} Contains {m['contains']} Refusals {m['refusals']}", flush=True)
        items = DATA["verified"]; jobs = []
        for n, x in enumerate(items):
            ctxs = {"closed_book": ""}
            if x["top5"]: ctxs["retrieved"] = "\n\n".join(f"[{i+1}] {t[:400]}" for i, t in enumerate(x["top5"]))
            ctxs["verified"] = window(x["verified_passage"], x["gold"])
            for c, ctx in ctxs.items(): jobs.append((n, x, c, pilot_prompt(x["lang"], x["query"], ctx)))
        preds = generate(tok, model, [j[3] for j in jobs]); rows = []
        for (n, x, c, _), p in zip(jobs, preds): rows.append(dict(n=n, lang=x["lang"], cond=c, pred=p, **score(p, x["gold"])))
        res["pilot"]["rows"] = rows
        for L in ("ar", "ms"):
            for c in ("closed_book", "retrieved", "verified"):
                r = [x for x in rows if x["lang"] == L and x["cond"] == c]
                print(f"[pilot {L}] {c:11s} n={len(r)} contains={sum(x['contains'] for x in r)} EM={sum(x['em'] for x in r)} refusals={sum(x['refusal'] for x in r)}")
        same = {(x["lang"], x["n"]) for x in rows if x["cond"] == "retrieved"}
        for c in ("closed_book", "retrieved", "verified"):
            r = [x for x in rows if x["cond"] == c and (x["lang"], x["n"]) in same]
            print(f"[pilot matched 29] {c:11s} n={len(r)} contains={sum(x['contains'] for x in r)} EM={sum(x['em'] for x in r)} refusals={sum(x['refusal'] for x in r)}")
        json.dump(res, open(f"results_second_gen/{short}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"saved results_second_gen/{short}.json  ({time.time()-t0:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache()
    print("DONE")


main()
