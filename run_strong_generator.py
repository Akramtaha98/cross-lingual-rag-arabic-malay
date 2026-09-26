"""
run_strong_generator.py  —  Strong Generator Baseline (M4 Pro / Apple Silicon)
===============================================================================
Runs Qwen2.5-7B-Instruct as a RAG generator on 200 MKQA samples per language
and scores with BERTScore + LLM-as-judge (claude-haiku-4-5).

Fills Table 8 (strong generator comparison) in the paper.

RECOMMENDED: ollama backend (fastest on M4 Pro, ~35-50 tok/s with Metal)
FALLBACK:    HuggingFace transformers with MPS (~12-18 tok/s)

========================================================
  SETUP (one-time, ~10 minutes)
========================================================

Option A — ollama (RECOMMENDED for M4 Pro):
  1. brew install ollama          (or download from https://ollama.ai)
  2. ollama serve                 (keep this terminal open)
  3. ollama pull qwen2.5:7b       (downloads ~4.7 GB, Q4_K_M quantized)
  Then run:
     python run_strong_generator.py --backend ollama --lang ar --n 200 --judge
     python run_strong_generator.py --backend ollama --lang ms --n 200 --judge

Option B — HuggingFace transformers + MPS:
  pip install transformers accelerate bert-score anthropic --break-system-packages
  Then run:
     python run_strong_generator.py --backend hf --lang ar --n 200
     python run_strong_generator.py --backend hf --lang ms --n 200

========================================================
  EXPECTED TIME ON M4 PRO (24 GB unified memory)
========================================================
  Retrieval encoding (mE5-large, 200 queries):   ~2 min per language
  Ollama inference (200 samples × ~80 tokens):   ~10-15 min per language
  HF-MPS inference (200 samples × ~80 tokens):   ~30-40 min per language
  LLM judge (50 samples via API):                ~3-4 min per language
  ─────────────────────────────────────────────────────
  Total with ollama (both languages):            ~35-45 minutes
  Total with HF-MPS (both languages):            ~80-90 minutes

========================================================
  OUTPUTS
========================================================
  results/strong_gen_<backend>_<lang>_results.json
  Console: ready-to-paste LaTeX row for Table 8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
RESULTS_DIR  = SCRIPT_DIR / "results"
INDEXES_DIR  = SCRIPT_DIR / "indexes"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Prompt templates ─────────────────────────────────────────────────────────
def make_prompt(query: str, context: str, lang: str) -> str:
    if lang == "ar":
        return (
            "أنت مساعد مفيد يجيب على الأسئلة باللغة العربية.\n"
            "استخدم فقط المقاطع المقدمة للإجابة. إذا لم تكن الإجابة في المقاطع، قل «لا أعلم».\n\n"
            f"المقاطع:\n{context}\n\n"
            f"السؤال: {query}\n"
            "الإجابة:"
        )
    else:
        return (
            "Anda adalah pembantu yang berguna yang menjawab soalan dalam Bahasa Melayu.\n"
            "Gunakan hanya petikan yang diberikan untuk menjawab. Jika jawapan tiada dalam petikan, katakan «Tidak tahu».\n\n"
            f"Petikan:\n{context}\n\n"
            f"Soalan: {query}\n"
            "Jawapan:"
        )

LLM_JUDGE_PROMPT = """\
You are evaluating a question-answering system.

Question: {query}
Reference answer: {reference}
System answer: {answer}

Rate on TWO criteria (1–5 scale):
1. Faithfulness: Is the answer grounded in the retrieved context? (1=hallucinated, 5=fully grounded)
2. Adequacy: Does the answer correctly address the question? (1=wrong/irrelevant, 5=fully correct)

Respond ONLY as JSON: {{"faithfulness": <int>, "adequacy": <int>, "comment": "<one sentence>"}}"""


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_mkqa(lang: str, n: int) -> list:
    """Download MKQA from apple/ml-mkqa and parse correctly."""
    import gzip, ssl, urllib.request

    # Use the project's own cached copy if it exists
    project_cache = SCRIPT_DIR / "indexes" / "mkqa.jsonl.gz"
    tmp_cache     = Path("/tmp/mkqa.jsonl.gz")
    cache_path    = project_cache if project_cache.exists() else tmp_cache

    if not cache_path.exists():
        urls = [
            "https://raw.githubusercontent.com/apple/ml-mkqa/main/mkqa.jsonl.gz",
            "https://raw.githubusercontent.com/apple/ml-mkqa/master/mkqa.jsonl.gz",
        ]
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        downloaded = False
        for url in urls:
            try:
                print(f"Downloading MKQA from {url} ...")
                with urllib.request.urlopen(url, context=ctx) as r:
                    cache_path.write_bytes(r.read())
                print("Download complete.")
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed ({url}): {e}")
        if not downloaded:
            raise RuntimeError(
                "Could not download MKQA.\n"
                "Manually download mkqa.jsonl.gz from "
                "https://github.com/apple/ml-mkqa and place it at:\n"
                f"  {project_cache}"
            )

    samples = []
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Queries are in item["queries"][lang], answers in item["answers"][lang]
            query = item.get("queries", {}).get(lang) or item.get("query", "")
            ans_list = item.get("answers", {}).get(lang, [])
            # ans_list is a list of dicts: [{"text": "...", "aliases": [...]}]
            if isinstance(ans_list, list) and ans_list:
                answer_text = ans_list[0].get("text", "") if isinstance(ans_list[0], dict) else str(ans_list[0])
            else:
                answer_text = ""
            if query and answer_text:
                samples.append({
                    "query":     query,
                    "reference": answer_text,
                    "lang":      lang,
                })
            if len(samples) >= n:
                break

    print(f"Loaded {len(samples)} MKQA samples for lang={lang}")
    return samples


def load_index_and_passages(lang: str):
    """Load the mE5-large FAISS index and passage list."""
    import faiss
    import json

    index_path    = INDEXES_DIR / f"me5large_{lang}.faiss"
    passages_path = INDEXES_DIR / f"me5large_{lang}_passages.json"

    if not index_path.exists():
        print(f"ERROR: FAISS index not found: {index_path}")
        sys.exit(1)

    index = faiss.read_index(str(index_path))
    with open(passages_path, encoding="utf-8") as f:
        passages = json.load(f)   # list of dicts: passage_id, text, title, lang

    print(f"Loaded FAISS index ({index.ntotal} vectors) + {len(passages)} passages")
    return index, passages


def retrieve(query: str, index, passages: list,
             encode_fn, k: int = 5) -> list:
    """Return [(passage_dict, score), ...]."""
    import numpy as np
    vec = encode_fn([f"query: {query}"])
    scores, idxs = index.search(vec.astype("float32"), k)
    return [(passages[i], float(s)) for s, i in zip(scores[0], idxs[0])
            if 0 <= i < len(passages)]


def build_context(retrieved: list, max_chars: int = 2000) -> str:
    parts, total = [], 0
    for i, (p, _) in enumerate(retrieved):
        text = p.get("text", "")[:max_chars - total]
        parts.append(f"[{i+1}] {text}")
        total += len(text)
        if total >= max_chars:
            break
    return "\n\n".join(parts)


# ─── Retriever (mE5-large, Apple-Silicon friendly) ────────────────────────────

def load_retriever():
    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading mE5-large on {device} ...")
    model = SentenceTransformer("intfloat/multilingual-e5-large", device=device)
    def encode_fn(texts):
        return model.encode(texts, normalize_embeddings=True)
    return encode_fn


# ─── Generator backends ───────────────────────────────────────────────────────

# --- A. Ollama (recommended for M4 Pro) ---------------------------------------

def generate_ollama(prompt: str, model_name: str = "qwen2.5:7b",
                    max_tokens: int = 120) -> str:
    """Call local ollama server (must be running: `ollama serve`)."""
    import urllib.request
    payload = json.dumps({
        "model":  model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result.get("response", "").strip()
    except Exception as e:
        return f"[ollama error: {e}]"


# --- B. HuggingFace transformers + MPS ----------------------------------------

def load_hf_model(model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {model_id} on {device} (this downloads ~14 GB the first time) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,   # float16 works on MPS; bfloat16 may error
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    print("Model ready.")
    return tokenizer, model, device


def generate_hf(prompt: str, tokenizer, model, device: str,
                max_new_tokens: int = 120) -> str:
    import torch
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ─── Evaluation ───────────────────────────────────────────────────────────────

def compute_bertscore(predictions: list, references: list) -> dict:
    from bert_score import score as bscore
    P, R, F = bscore(predictions, references,
                     model_type="bert-base-multilingual-cased",
                     lang="other", verbose=False)
    return {"BERTScore_P": float(P.mean()),
            "BERTScore_R": float(R.mean()),
            "BERTScore_F": float(F.mean())}


def llm_judge(samples: list, n: int = 50) -> dict:
    """Score n samples with claude-haiku-4-5 via Anthropic API."""
    import anthropic, statistics
    client = anthropic.Anthropic()
    faith_scores, adeq_scores, annotations = [], [], []

    for o in samples[:n]:
        prompt = LLM_JUDGE_PROMPT.format(
            query=o["query"], reference=o["reference"], answer=o["answer"])
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5", max_tokens=200,
                messages=[{"role": "user", "content": prompt}])
            raw = msg.content[0].text.strip()
            parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
            faith = int(parsed.get("faithfulness", 1))
            adeq  = int(parsed.get("adequacy",     1))
            comment = parsed.get("comment", "")
        except Exception as e:
            faith, adeq, comment = 1, 1, str(e)
        faith_scores.append(faith)
        adeq_scores.append(adeq)
        annotations.append({**o, "faithfulness": faith,
                             "adequacy": adeq, "comment": comment})
        time.sleep(0.25)

    return {
        "Faithfulness_mean": statistics.mean(faith_scores),
        "Faithfulness_std":  statistics.stdev(faith_scores) if len(faith_scores) > 1 else 0.0,
        "Adequacy_mean":     statistics.mean(adeq_scores),
        "Adequacy_std":      statistics.stdev(adeq_scores) if len(adeq_scores) > 1 else 0.0,
        "annotations":       annotations,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ollama",
                        choices=["ollama", "hf"],
                        help="ollama (recommended, Metal-accelerated) or hf (MPS)")
    parser.add_argument("--model",   default="qwen2.5:7b",
                        help="ollama: 'qwen2.5:7b' | hf: 'Qwen/Qwen2.5-7B-Instruct'")
    parser.add_argument("--lang",    default="ar", choices=["ar", "ms"])
    parser.add_argument("--n",       type=int, default=200)
    parser.add_argument("--k",       type=int, default=5)
    parser.add_argument("--judge",   action="store_true",
                        help="Run LLM-as-judge (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--judge-n", type=int, default=50)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Strong Generator Experiment")
    print(f"  Backend : {args.backend}  |  Model: {args.model}")
    print(f"  Lang    : {args.lang}  |  N: {args.n}  |  K: {args.k}")
    print(f"{'='*55}\n")

    # 1. Load MKQA data
    samples = load_mkqa(args.lang, args.n)

    # 2. Load retrieval index + encode function (mE5-large)
    encode_fn = load_retriever()
    index, passages = load_index_and_passages(args.lang)

    # 3. Load generator
    if args.backend == "hf":
        hf_model_id = (args.model if "/" in args.model
                       else "Qwen/Qwen2.5-7B-Instruct")
        tokenizer, gen_model, device = load_hf_model(hf_model_id)

    # 4. RAG inference loop
    outputs = []
    t_start = time.time()
    print(f"Running RAG on {len(samples)} samples ...\n")

    for i, s in enumerate(samples):
        retrieved = retrieve(s["query"], index, passages, encode_fn, k=args.k)
        context   = build_context(retrieved)
        prompt    = make_prompt(s["query"], context, args.lang)

        if args.backend == "ollama":
            answer = generate_ollama(prompt, model_name=args.model)
        else:
            answer = generate_hf(prompt, tokenizer, gen_model, device)

        outputs.append({
            "query":     s["query"],
            "reference": s["reference"],
            "answer":    answer,
            "lang":      args.lang,
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed
            eta     = (len(samples) - i - 1) / rate
            print(f"  [{i+1}/{len(samples)}]  {rate:.1f} sample/s  "
                  f"ETA {eta/60:.1f} min")
            print(f"  Last answer: {answer[:90]}")

    elapsed_inf = time.time() - t_start
    print(f"\nInference done in {elapsed_inf/60:.1f} min "
          f"({elapsed_inf/len(outputs):.1f}s/sample)")

    # 5. BERTScore
    print("\nComputing BERTScore ...")
    preds = [o["answer"]    for o in outputs]
    refs  = [o["reference"] for o in outputs]
    bs    = compute_bertscore(preds, refs)
    print(f"BERTScore-F: {bs['BERTScore_F']:.4f}")

    # 6. LLM judge (optional)
    judge_metrics = {}
    if args.judge:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("WARNING: ANTHROPIC_API_KEY not set — skipping judge")
        else:
            print(f"\nRunning LLM judge on {args.judge_n} samples ...")
            judge_metrics = llm_judge(outputs, n=args.judge_n)
            print(f"Faithfulness: {judge_metrics['Faithfulness_mean']:.2f} "
                  f"± {judge_metrics['Faithfulness_std']:.2f}")
            print(f"Adequacy:     {judge_metrics['Adequacy_mean']:.2f} "
                  f"± {judge_metrics['Adequacy_std']:.2f}")

    # 7. Save
    model_short = args.model.replace("/", "_").replace(":", "_")
    out_file = RESULTS_DIR / f"strong_gen_{model_short}_{args.lang}_results.json"
    result = {
        "config":    {"backend": args.backend, "model": args.model,
                      "lang": args.lang, "n": len(outputs), "k": args.k},
        "metrics":   {**bs, **{k: v for k, v in judge_metrics.items()
                                if k != "annotations"}},
        "samples":   outputs[:10],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_file}")

    # 8. Print LaTeX row for Table 8
    bsf   = bs["BERTScore_F"]
    faith = judge_metrics.get("Faithfulness_mean", "—")
    adeq  = judge_metrics.get("Adequacy_mean",     "—")
    f_str = f"{faith:.2f}" if isinstance(faith, float) else faith
    a_str = f"{adeq:.2f}"  if isinstance(adeq,  float) else adeq
    lang_label = args.lang.upper()
    short_name = args.model.split(":")[0].replace("qwen2.5", "Qwen2.5-7B")
    print(f"\n{'='*55}")
    print("  Paste this row into Table 8 in cross_lingual_rag.tex:")
    print(f"  {short_name} (RAG) & {lang_label} & {bsf:.4f} & {f_str} & {a_str} \\\\")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
