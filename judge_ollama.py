#!/usr/bin/env python3
"""
Free, local LLM-judge using Ollama instead of the Anthropic API.
Re-scores an existing run_strong_generator.py / run_matched_retriever_gen.py
results JSON in place (adds/overwrites "judge_metrics" and per-sample
"faithfulness"/"adequacy"/"comment" fields), using a local Ollama model
as judge. No API key, no cost.

Usage:
    python judge_ollama.py --file results/strong_gen_qwen2.5_7b_ar_results.json
    python judge_ollama.py --file results/strong_gen_qwen2.5_7b_ms_results.json
    python judge_ollama.py --file results/table9_matched_me5large_mt5_base_ar_results.json
    ... (repeat for each results file you want judged)

Requires `ollama serve` running (same as generation).
By construction the judge model differs from the system under evaluation
whenever possible: for Qwen2.5-7B outputs, judging with a *different*
local model (e.g. llama3.1:8b if pulled) avoids a same-model self-judge
bias; for mT5 outputs, Qwen2.5-7B is a fine, independent judge.
"""
import argparse
import json
import statistics
import time
import urllib.request

JUDGE_PROMPT = """\
You are evaluating a question-answering system.

Question: {query}
Reference answer: {reference}
System answer: {answer}

Rate on TWO criteria (1-5 scale):
1. Faithfulness: Is the answer grounded in the retrieved context? (1=hallucinated, 5=fully grounded)
2. Adequacy: Does the answer correctly address the question? (1=wrong/irrelevant, 5=fully correct)

Respond ONLY as JSON: {{"faithfulness": <int>, "adequacy": <int>, "comment": "<one sentence>"}}"""


def call_ollama(prompt: str, model: str, timeout: int = 120) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()).get("response", "").strip()


def judge_samples(samples, model, n):
    faith_scores, adeq_scores, annotations = [], [], []
    n_errors = 0
    for i, o in enumerate(samples[:n]):
        prompt = JUDGE_PROMPT.format(
            query=o.get("query", ""),
            reference=o.get("reference", ""),
            answer=o.get("answer", ""),
        )
        try:
            raw = call_ollama(prompt, model)
            parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            faith = int(parsed.get("faithfulness", 1))
            adeq = int(parsed.get("adequacy", 1))
            comment = parsed.get("comment", "")
        except Exception as e:
            faith, adeq, comment = 1, 1, f"[judge error: {e}]"
            n_errors += 1
        faith_scores.append(faith)
        adeq_scores.append(adeq)
        annotations.append({**o, "faithfulness": faith, "adequacy": adeq, "comment": comment})
        if (i + 1) % 10 == 0:
            print(f"  judged {i + 1}/{min(n, len(samples))}  (errors so far: {n_errors})")
    return {
        "Faithfulness_mean": statistics.mean(faith_scores),
        "Faithfulness_std": statistics.stdev(faith_scores) if len(faith_scores) > 1 else 0.0,
        "Adequacy_mean": statistics.mean(adeq_scores),
        "Adequacy_std": statistics.stdev(adeq_scores) if len(adeq_scores) > 1 else 0.0,
        "n_judged": len(faith_scores),
        "n_errors": n_errors,
        "judge_model": model,
        "annotations": annotations,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="results JSON to judge (in place)")
    ap.add_argument("--judge-model", default="qwen2.5:7b",
                    help="Ollama model to use as judge (pull a different model than the "
                         "system under test when possible, e.g. llama3.1:8b, to avoid self-judging bias)")
    ap.add_argument("--n", type=int, default=50, help="number of samples to judge")
    args = ap.parse_args()

    with open(args.file) as f:
        data = json.load(f)

    samples = data.get("samples")
    if not samples:
        raise SystemExit(f"No 'samples' key found in {args.file} — nothing to judge.")

    print(f"Judging {min(args.n, len(samples))} samples from {args.file} "
          f"with local model '{args.judge_model}' ...")
    t0 = time.time()
    judge_metrics = judge_samples(samples, args.judge_model, args.n)
    dt = time.time() - t0

    data.setdefault("metrics", {})
    data["metrics"]["Faithfulness_mean"] = judge_metrics["Faithfulness_mean"]
    data["metrics"]["Faithfulness_std"] = judge_metrics["Faithfulness_std"]
    data["metrics"]["Adequacy_mean"] = judge_metrics["Adequacy_mean"]
    data["metrics"]["Adequacy_std"] = judge_metrics["Adequacy_std"]
    data["judge_meta"] = {
        "backend": "ollama",
        "judge_model": args.judge_model,
        "n_judged": judge_metrics["n_judged"],
        "n_errors": judge_metrics["n_errors"],
    }
    data["judge_annotations"] = judge_metrics["annotations"]

    with open(args.file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDone in {dt/60:.1f} min.")
    print(f"Faithfulness: {judge_metrics['Faithfulness_mean']:.2f} ± {judge_metrics['Faithfulness_std']:.2f}")
    print(f"Adequacy:     {judge_metrics['Adequacy_mean']:.2f} ± {judge_metrics['Adequacy_std']:.2f}")
    print(f"Errors: {judge_metrics['n_errors']}/{judge_metrics['n_judged']}")
    print(f"Written back to {args.file}")


if __name__ == "__main__":
    main()
