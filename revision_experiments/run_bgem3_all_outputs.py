#!/usr/bin/env python3
"""BGE-M3 dense vs sparse vs multi-vector (ColBERT) vs combined retrieval on the same 1,000 questions,
the same 50,000-passage corpora and the same answer-string relevance proxy as Table 2.

Run from the repo root on the H200 (GPU 0):
    CUDA_VISIBLE_DEVICES=0 python3 revision_experiments/run_bgem3_all_outputs.py --n 20     # smoke test
    CUDA_VISIBLE_DEVICES=0 python3 revision_experiments/run_bgem3_all_outputs.py            # full run
Needs in indexes/: bgem3_{ar,ms}_passages.json and mkqa.jsonl.gz (copy them from the Mac; see HOW_TO below).
Needs packages: torch, transformers, huggingface_hub, scipy, numpy (no FlagEmbedding needed; the three M3 heads are
re-implemented exactly as in FlagEmbedding: dense = normalized [CLS]; sparse = relu(linear(h)) with max weight per token
id; multi-vector = normalized linear(h[1:]) with MaxSim averaged over query tokens).
Output: results/bgem3_all_outputs.json

Variants (top-10 lists; Hit@5, Hit@10, MRR, fractional Recall@5, 2,000-resample percentile bootstrap):
  dense                 full-corpus dense search
  sparse                full-corpus lexical-weight search
  dense+sparse          1.0*dense + 0.3*sparse, reranking the union of the dense and sparse top-30
  multi-vector          ColBERT MaxSim, reranking the same union
  all (0.4/0.2/0.4)     M3 default weights, reranking the same union
"""
import argparse, gzip, json, time
import numpy as np, torch, scipy.sparse as sp
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--langs", default="ar,ms")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--depth", type=int, default=30)
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--bs", type=int, default=64)
args = ap.parse_args()
dev = "cuda"
TOP = 10

tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
enc = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).to(dev).eval()
H = enc.config.hidden_size
colbert = torch.nn.Linear(H, H).to(dev).half().eval()
sparse = torch.nn.Linear(H, 1).to(dev).half().eval()
colbert.load_state_dict(torch.load(hf_hub_download("BAAI/bge-m3", "colbert_linear.pt"), map_location=dev))
sparse.load_state_dict(torch.load(hf_hub_download("BAAI/bge-m3", "sparse_linear.pt"), map_location=dev))
UNUSED = {tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id}


@torch.no_grad()
def forward(texts, want_colbert=False):
    b = tok(texts, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to(dev)
    h = enc(**b).last_hidden_state
    dense = torch.nn.functional.normalize(h[:, 0].float(), dim=-1)
    w = torch.relu(sparse(h)).squeeze(-1).float()            # B x L
    ids = b["input_ids"]
    sp_rows = []
    for i in range(len(texts)):
        d = {}
        for t, x in zip(ids[i].tolist(), w[i].tolist()):
            if t in UNUSED or x <= 0: continue
            if x > d.get(t, 0): d[t] = x
        sp_rows.append(d)
    cols = None
    if want_colbert:
        c = colbert(h[:, 1:]).float() * b["attention_mask"][:, 1:, None].float()
        c = torch.nn.functional.normalize(c, dim=-1)
        cols = [c[i, : int(b["attention_mask"][i, 1:].sum())] for i in range(len(texts))]
    return dense, sp_rows, cols


def encode_corpus(texts):
    order = np.argsort([-len(t) for t in texts]); D = np.zeros((len(texts), H), dtype=np.float32); S = [None] * len(texts)
    for s in range(0, len(texts), args.bs):
        idx = order[s:s + args.bs]; d, sr, _ = forward([texts[i] for i in idx])
        D[idx] = d.cpu().numpy()
        for j, i in enumerate(idx): S[i] = sr[j]
        if (s // args.bs) % 100 == 0: print(f"  corpus {s}/{len(texts)}", flush=True)
    rows, cols, vals = [], [], []
    for i, d in enumerate(S):
        for t, x in d.items(): rows.append(i); cols.append(t); vals.append(x)
    M = sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(tok)), dtype=np.float32)
    return D, M


def load_eval(lang, n):
    rows = []
    with gzip.open("indexes/mkqa.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); q = r.get("queries", {}).get(lang)
            if not q: continue
            a = r.get("answers", {}).get(lang) or []
            rows.append({"query": q, "answer": (a[0].get("text", "") or "") if a else ""})
            if len(rows) >= n: break
    return rows


def metrics_for(ranked, relevant):
    hits = [1 if i in relevant else 0 for i in ranked]
    rr = next((1.0 / (k + 1) for k, h in enumerate(hits) if h), 0.0)
    return float(any(hits[:5])), float(any(hits[:10])), rr, (sum(hits[:5]) / len(relevant)) if relevant else 0.0


def boot_ci(x, B, rng):
    x = np.asarray(x); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def topk(scores, k):
    t = np.argpartition(-scores, k)[:k]; return t[np.lexsort((t, -scores[t]))]


def main():
    rng = np.random.default_rng(0); out = {}
    for lang in args.langs.split(","):
        t0 = time.time()
        items = load_eval(lang, args.n)
        texts = [p["text"] for p in json.load(open(f"indexes/bgem3_{lang}_passages.json", encoding="utf-8"))]
        print(f"[{lang}] {len(items)} questions, {len(texts)} passages", flush=True)
        D, M = encode_corpus(texts); print(f"[{lang}] corpus encoded {time.time()-t0:.0f}s", flush=True)
        Dg = torch.from_numpy(D).to(dev)
        low = [t.lower() for t in texts]
        rel = []
        for it in items:
            a = it["answer"].strip().lower(); rel.append({i for i, t in enumerate(low) if a in t} if a else set())
        runs = {k: [] for k in ["dense", "sparse", "dense+sparse", "multi-vector", "all (0.4/0.2/0.4)"]}
        for qi, it in enumerate(items):
            qd, qs, qc = forward([it["query"]], want_colbert=True)
            ds = (Dg @ qd[0]).cpu().numpy()
            qvec = sp.csr_matrix(([v for v in qs[0].values()], ([0] * len(qs[0]), list(qs[0].keys()))), shape=(1, len(tok)), dtype=np.float32)
            ss = (M @ qvec.T).toarray().ravel()
            runs["dense"].append([int(i) for i in topk(ds, TOP)]); runs["sparse"].append([int(i) for i in topk(ss, TOP)])
            cand = sorted(set(topk(ds, args.depth).tolist()) | set(topk(ss, args.depth).tolist()))
            cs = np.zeros(len(cand), dtype=np.float32)
            for s in range(0, len(cand), args.bs):
                _, _, pc = forward([texts[i] for i in cand[s:s + args.bs]], want_colbert=True)
                for j, p in enumerate(pc):
                    cs[s + j] = float((qc[0] @ p.T).max(-1).values.sum() / qc[0].shape[0])
            d_c, s_c = ds[cand], ss[cand]
            for name, sc in (("dense+sparse", 1.0 * d_c + 0.3 * s_c), ("multi-vector", cs), ("all (0.4/0.2/0.4)", 0.4 * d_c + 0.2 * s_c + 0.4 * cs)):
                o = np.lexsort((np.array(cand), -sc))[:TOP]; runs[name].append([int(cand[i]) for i in o])
            if (qi + 1) % 100 == 0: print(f"  [{lang}] {qi+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
        per = {k: np.array([metrics_for(r, rel[q]) for q, r in enumerate(v)]) for k, v in runs.items()}
        names = ["Hit@5", "Hit@10", "MRR", "Recall@5"]
        res = {"n": len(items), "systems": {}, "paired_vs_dense": {}, "mean_candidates_note": f"reranked variants use the union of dense and sparse top-{args.depth}"}
        for k, arr in per.items():
            res["systems"][k] = {m: {"mean": float(arr[:, j].mean()), "ci": boot_ci(arr[:, j], args.boot, rng)} for j, m in enumerate(names)}
        for k in per:
            if k == "dense": continue
            d = per[k] - per["dense"]
            res["paired_vs_dense"][k] = {m: {"mean": float(d[:, j].mean()), "ci": boot_ci(d[:, j], args.boot, rng)} for j, m in enumerate(names)}
        out[lang] = res
        print(f"\n=== {lang} (n={len(items)}) ===")
        for k, v in res["systems"].items():
            print(f"{k:20s} " + "  ".join(f"{m} {v[m]['mean']:.3f} [{v[m]['ci'][0]:.3f},{v[m]['ci'][1]:.3f}]" for m in names))
        for k, v in res["paired_vs_dense"].items():
            print(f"  diff {k:20s} " + "  ".join(f"{m} {v[m]['mean']:+.3f} [{v[m]['ci'][0]:+.3f},{v[m]['ci'][1]:+.3f}]" for m in ("Hit@5", "MRR")))
    json.dump(out, open("results/bgem3_all_outputs.json", "w"), indent=1)
    print("\nSaved results/bgem3_all_outputs.json\nCHECK: the dense row should be close to Table 2 BGE-M3 (Hit@5 about .067 Arabic, .068 Malay).")


main()
