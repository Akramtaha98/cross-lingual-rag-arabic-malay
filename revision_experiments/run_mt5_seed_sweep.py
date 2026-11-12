#!/usr/bin/env python3
"""Designed multi-seed sweep for mT5-small (base vs LoRA) with LaBSE top-5 retrieval, 1,000 items per language.

For each seed S: (1) retrain the LoRA adapter with seed S (same hyper-parameters as train_lora.py, training items
1,001+ of MKQA ar+ms), (2) decode LoRA and base mT5-small on the 1,000 evaluation questions per language with the
sampling configuration of config.py and decoding seed S. Reports per-seed BERTScore-F, ROUGE-L, BLEU-4, the
LoRA-minus-base difference per seed, and mean +/- SD and a t-based 95% interval over seeds.

Run from the repo root on the H200 (GPU 0):
    CUDA_VISIBLE_DEVICES=0 python3 revision_experiments/run_mt5_seed_sweep.py --dev          # smoke test, ~3 min
    CUDA_VISIBLE_DEVICES=0 python3 revision_experiments/run_mt5_seed_sweep.py                # full, seeds 1-5
Resumable: finished (seed, language) results are stored in results/seed_sweep/ and skipped on rerun.
Needs in indexes/: labse_{ar,ms}_passages.json and mkqa.jsonl.gz (copy them from the Mac).
Output: results/seed_sweep/seed{S}_{lang}.json and results/seed_sweep_summary.json
"""
import argparse, gzip, json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.getcwd())
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--dev", action="store_true", help="1 seed, 1 epoch, 100 training pairs, 50 eval items")
args = ap.parse_args()
if args.dev: args.seeds, args.n = [1], 50

import train_lora as TL
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import generation.mt5_generator as G
from evaluation.metrics import compute_bleu, compute_rouge_l
from sentence_transformers import SentenceTransformer

OUT = "results/seed_sweep"; os.makedirs(OUT, exist_ok=True); dev = "cuda"


def load_eval(lang, n):
    rows = []
    with gzip.open("indexes/mkqa.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); q = r.get("queries", {}).get(lang)
            if not q: continue
            a = r.get("answers", {}).get(lang) or []
            rows.append({"query": q, "reference": (a[0].get("text", "") or "") if a else ""})
            if len(rows) >= n: break
    return rows


def labse_retrieve():
    cache = {}
    enc = SentenceTransformer("sentence-transformers/LaBSE", device=dev)
    for lang in ("ar", "ms"):
        items = load_eval(lang, args.n)
        passages = json.load(open(f"indexes/labse_{lang}_passages.json", encoding="utf-8"))
        P = enc.encode([p["text"] for p in passages], batch_size=128, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        Q = enc.encode([it["query"] for it in items], batch_size=128, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        top = (Q @ P.T).topk(5, dim=1).indices.cpu().tolist()
        cache[lang] = (items, [[(passages[j], 1.0) for j in row] for row in top])
        print(f"[retrieval] {lang}: {len(items)} queries, {len(passages)} passages", flush=True)
    del enc; torch.cuda.empty_cache(); return cache


def collate(tokenizer, batch):
    ins, tgt = zip(*batch)
    mi = tokenizer(list(ins), max_length=TL.MAX_INPUT_LENGTH, padding=True, truncation=True, return_tensors="pt")
    lab = tokenizer(text_target=list(tgt), max_length=TL.MAX_TARGET_LENGTH, padding=True, truncation=True, return_tensors="pt").input_ids
    lab[lab == tokenizer.pad_token_id] = -100; mi["labels"] = lab; return mi


def train_lora_seed(seed, path):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    pairs = TL.build_pairs(max_samples=100 if args.dev else None)
    tokenizer = AutoTokenizer.from_pretrained(TL.BASE_MODEL_NAME)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TL.MKQADataset(pairs), batch_size=TL.BATCH_SIZE, shuffle=True, generator=gen, collate_fn=lambda b: collate(tokenizer, b))
    model = AutoModelForSeq2SeqLM.from_pretrained(TL.BASE_MODEL_NAME)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=TL.LORA_R, lora_alpha=TL.LORA_ALPHA, lora_dropout=TL.LORA_DROPOUT, target_modules=TL.TARGET_MODULES, bias="none")).to(dev).train()
    epochs = 1 if args.dev else TL.NUM_EPOCHS
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=TL.LEARNING_RATE)
    total = max(1, (len(loader) // TL.GRAD_ACCUM) * epochs)
    sched = get_linear_schedule_with_warmup(opt, min(TL.WARMUP_STEPS, total // 10), total)
    opt.zero_grad()
    for ep in range(epochs):
        tot = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss = model(**batch).loss; (loss / TL.GRAD_ACCUM).backward(); tot += loss.item()
            if (step + 1) % TL.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
        print(f"[train seed {seed}] epoch {ep+1}/{epochs} avg loss {tot/len(loader):.4f}", flush=True)
    os.makedirs(path, exist_ok=True); model.save_pretrained(path); tokenizer.save_pretrained(path)
    del model; torch.cuda.empty_cache()


def bert_f(preds, refs, lang):
    from bert_score import score as bs
    keep = [i for i, p in enumerate(preds) if str(p).strip()]
    f = np.zeros(len(preds), dtype=np.float32)          # empty output scores 0, as in the second-generator study
    if keep:
        _, _, F = bs([preds[i] for i in keep], [refs[i] for i in keep], model_type="bert-base-multilingual-cased", lang=lang, verbose=False)
        f[keep] = F.numpy()
    return f


def decode(use_lora, lora_path, seed, items, retrieved):
    G.LORA_MODEL_PATH = lora_path
    gen = G.MT5Generator(use_lora=use_lora, device=dev)
    gen._base_seed = seed * 100000                      # per-call seeding as in the main experiments, offset by the sweep seed
    preds = [gen.generate(it["query"], r, top_k=5) for it, r in zip(items, retrieved)]
    del gen; torch.cuda.empty_cache(); return preds


def main():
    cache = labse_retrieve()
    for seed in args.seeds:
        todo = [l for l in ("ar", "ms") if not os.path.exists(f"{OUT}/seed{seed}_{l}.json")]
        if not todo: continue
        path = f"models/mt5_lora_seed{seed}"
        if not os.path.exists(path): t0 = time.time(); train_lora_seed(seed, path); print(f"[seed {seed}] trained in {time.time()-t0:.0f}s", flush=True)
        for lang in todo:
            items, retrieved = cache[lang]; refs = [it["reference"] for it in items]; res = {"seed": seed, "lang": lang, "n": len(items)}
            for name, use in (("lora", True), ("base", False)):
                preds = decode(use, path, seed, items, retrieved)
                f = bert_f(preds, refs, lang)
                res[name] = {"BERTScore_F": float(f.mean()), "BLEU-4": compute_bleu(preds, refs), "ROUGE-L": compute_rouge_l(preds, refs),
                             "empty_outputs": int(sum(1 for p in preds if not str(p).strip())), "bert_f_items": f.tolist(), "samples": preds[:10]}
            res["delta_BERTScore_F"] = res["lora"]["BERTScore_F"] - res["base"]["BERTScore_F"]
            json.dump(res, open(f"{OUT}/seed{seed}_{lang}.json", "w", encoding="utf-8"), ensure_ascii=False)
            print(f"[seed {seed}] {lang}: LoRA BERT-F {res['lora']['BERTScore_F']:.4f}  base {res['base']['BERTScore_F']:.4f}  delta {res['delta_BERTScore_F']:+.4f}", flush=True)
    from scipy import stats
    summ = {}
    for lang in ("ar", "ms"):
        rs = [json.load(open(f"{OUT}/seed{s}_{lang}.json")) for s in args.seeds if os.path.exists(f"{OUT}/seed{s}_{lang}.json")]
        s = {"seeds": [r["seed"] for r in rs]}
        for m in ("BERTScore_F", "BLEU-4", "ROUGE-L"):
            for v in ("lora", "base"):
                x = np.array([r[v][m] for r in rs]); s[f"{v}_{m}"] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "values": x.tolist()}
        d = np.array([r["delta_BERTScore_F"] for r in rs]); s["delta_BERTScore_F"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0, "values": d.tolist()}
        if len(d) > 1:
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)); s["delta_BERTScore_F"]["ci95_over_seeds"] = [float(d.mean() - h), float(d.mean() + h)]
        summ[lang] = s
        print(f"\n== {lang} ({len(rs)} seeds) ==")
        for m in ("BERTScore_F", "BLEU-4", "ROUGE-L"):
            print(f"{m:12s} LoRA {s['lora_'+m]['mean']:.4f} +/- {s['lora_'+m]['sd']:.4f}   base {s['base_'+m]['mean']:.4f} +/- {s['base_'+m]['sd']:.4f}")
        print(f"delta BERT-F {s['delta_BERTScore_F']['mean']:+.4f} +/- {s['delta_BERTScore_F']['sd']:.4f}  95% CI over seeds {s['delta_BERTScore_F'].get('ci95_over_seeds')}")
    json.dump(summ, open("results/seed_sweep_summary.json", "w"), indent=1)
    print("\nSaved results/seed_sweep_summary.json")


main()
