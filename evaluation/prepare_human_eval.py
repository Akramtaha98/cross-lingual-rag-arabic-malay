"""
evaluation/prepare_human_eval.py
=================================
Prepare human evaluation annotation materials from experiment results.

Exports a stratified sample (default: 50 items per language) as:
  1. CSV file  — for spreadsheet annotation (Google Sheets / Excel)
  2. HTML file — self-contained annotation form with radio buttons

Annotators rate each (query, context, answer) triple on two dimensions:
  - Faithfulness  : Is the answer supported by the retrieved context? (1–5)
  - Adequacy      : Does the answer address the question? (1–5)

Usage
-----
    cd cross_lingual_rag
    python evaluation/prepare_human_eval.py

Output
------
    ./human_eval/human_eval_ar.csv
    ./human_eval/human_eval_ms.csv
    ./human_eval/annotation_form_ar.html
    ./human_eval/annotation_form_ms.html
    ./human_eval/ANNOTATOR_INSTRUCTIONS.txt
"""

import csv
import json
import os
import random
from pathlib import Path
from typing import List, Dict

RESULTS_DIR  = Path("./results")
OUTPUT_DIR   = Path("./human_eval")
SAMPLE_SIZE  = 50    # items per language
RANDOM_SEED  = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_experiment_samples(lang: str) -> List[Dict]:
    """
    Collect all samples across experiments for a given language.
    Avoids duplicates by tracking (query, answer) pairs.
    Returns list of dicts with keys: exp_name, query, context, answer, lang.
    """
    all_items = []
    seen = set()

    for path in sorted(RESULTS_DIR.glob("*_results.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        cfg = data.get("config", {})
        if cfg.get("query_lang") != lang:
            continue

        exp_name = path.stem.replace("_results", "")
        for sample in data.get("samples", []):
            query  = sample.get("query", "").strip()
            answer = sample.get("answer", "").strip()
            key = (query, answer)
            if key in seen or not query:
                continue
            seen.add(key)

            # Build readable context from passages
            passages = sample.get("passages", [])
            context_parts = []
            for i, item in enumerate(passages[:3], 1):
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    passage = item[0]
                else:
                    passage = item
                text = passage.get("text", "") if isinstance(passage, dict) else str(passage)
                context_parts.append(f"[{i}] {text[:300]}")
            context = "\n".join(context_parts)

            all_items.append({
                "exp_name": exp_name,
                "query":    query,
                "context":  context,
                "answer":   answer,
                "lang":     lang,
            })

    return all_items


def stratified_sample(items: List[Dict], n: int, seed: int = RANDOM_SEED) -> List[Dict]:
    """Sample n items, stratified by experiment to ensure coverage."""
    rng = random.Random(seed)
    from collections import defaultdict
    by_exp = defaultdict(list)
    for item in items:
        by_exp[item["exp_name"]].append(item)

    result = []
    exps = sorted(by_exp.keys())
    per_exp = max(1, n // len(exps))
    for exp in exps:
        pool = by_exp[exp]
        rng.shuffle(pool)
        result.extend(pool[:per_exp])

    # Top-up to n if needed
    remaining = [x for x in items if x not in result]
    rng.shuffle(remaining)
    result.extend(remaining[: max(0, n - len(result))])
    return result[:n]


# ── CSV export ────────────────────────────────────────────────────────────────

def write_csv(items: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id", "exp_name", "lang", "query",
        "context", "answer",
        "faithfulness_1_5", "adequacy_1_5", "annotator_comment"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, item in enumerate(items, 1):
            writer.writerow({
                "item_id":            i,
                "exp_name":           item["exp_name"],
                "lang":               item["lang"],
                "query":              item["query"],
                "context":            item["context"],
                "answer":             item["answer"],
                "faithfulness_1_5":   "",   # annotator fills this
                "adequacy_1_5":       "",
                "annotator_comment":  "",
            })
    print(f"  Wrote CSV: {path}")


# ── HTML annotation form ──────────────────────────────────────────────────────

LANG_LABELS = {
    "ar": {
        "title":       "Arabic QA Faithfulness Evaluation",
        "instr_query": "Question",
        "instr_ctx":   "Retrieved Context",
        "instr_ans":   "Generated Answer",
        "faith_label": "Faithfulness (1=not faithful, 5=fully faithful)",
        "adeq_label":  "Adequacy (1=does not answer, 5=fully answers the question)",
        "comment":     "Comment (optional)",
        "submit":      "Download Annotations (JSON)",
    },
    "ms": {
        "title":       "Malay QA Faithfulness Evaluation",
        "instr_query": "Question",
        "instr_ctx":   "Retrieved Context",
        "instr_ans":   "Generated Answer",
        "faith_label": "Faithfulness (1=tidak setia, 5=sangat setia)",
        "adeq_label":  "Kecukupan (1=tidak menjawab, 5=menjawab sepenuhnya)",
        "comment":     "Komen (pilihan)",
        "submit":      "Muat Turun Anotasi (JSON)",
    },
}


def write_html(items: List[Dict], path: Path, lang: str) -> None:
    lbl = LANG_LABELS.get(lang, LANG_LABELS["ar"])
    path.parent.mkdir(parents=True, exist_ok=True)

    cards_html = []
    for i, item in enumerate(items, 1):
        q   = item["query"].replace("&","&amp;").replace("<","&lt;")
        ctx = item["context"].replace("&","&amp;").replace("<","&lt;").replace("\n","<br>")
        ans = item["answer"].replace("&","&amp;").replace("<","&lt;")

        faith_radios = "".join(
            f'<label><input type="radio" name="faith_{i}" value="{v}" required> {v}</label> '
            for v in range(1, 6)
        )
        adeq_radios = "".join(
            f'<label><input type="radio" name="adeq_{i}" value="{v}" required> {v}</label> '
            for v in range(1, 6)
        )

        cards_html.append(f"""
<div class="card" id="item-{i}">
  <div class="item-num">Item {i} / {len(items)}</div>
  <div class="field">
    <label>{lbl['instr_query']}:</label>
    <div class="text-block query">{q}</div>
  </div>
  <div class="field">
    <label>{lbl['instr_ctx']}:</label>
    <div class="text-block ctx">{ctx}</div>
  </div>
  <div class="field">
    <label>{lbl['instr_ans']}:</label>
    <div class="text-block ans">{ans}</div>
  </div>
  <div class="rating">
    <label>{lbl['faith_label']}:</label><br>
    {faith_radios}
  </div>
  <div class="rating">
    <label>{lbl['adeq_label']}:</label><br>
    {adeq_radios}
  </div>
  <div class="field">
    <label>{lbl['comment']}:</label>
    <textarea name="comment_{i}" rows="2" style="width:100%"></textarea>
  </div>
</div>
""")

    # Embed item metadata as JS for the download function
    items_json = json.dumps(
        [{"item_id": i+1, "exp_name": x["exp_name"], "query": x["query"], "answer": x["answer"]}
         for i, x in enumerate(items)],
        ensure_ascii=False
    )

    html = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<title>{lbl['title']}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ color: #2c3e50; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0;
           background: #fafafa; }}
  .item-num {{ font-weight: bold; color: #7f8c8d; margin-bottom: 10px; }}
  .field {{ margin: 10px 0; }}
  .text-block {{ background: #fff; border: 1px solid #eee; padding: 10px; border-radius: 4px;
                 white-space: pre-wrap; font-size: 0.95em; }}
  .query  {{ color: #2980b9; font-weight: bold; }}
  .ctx    {{ color: #555; font-size: 0.9em; max-height: 150px; overflow-y: auto; }}
  .ans    {{ color: #27ae60; }}
  .rating {{ margin: 12px 0; }}
  label   {{ font-weight: bold; }}
  input[type=radio] {{ margin: 0 4px 0 8px; }}
  button  {{ background: #2980b9; color: white; padding: 12px 30px; border: none;
             border-radius: 6px; font-size: 1em; cursor: pointer; margin-top: 20px; }}
  button:hover {{ background: #1a6698; }}
  .progress {{ background: #ecf0f1; padding: 10px; border-radius: 4px; margin: 10px 0; }}
</style>
</head>
<body>
<h1>{lbl['title']}</h1>
<div class="progress" id="progress">
  Completed: <strong id="done">0</strong> / {len(items)}
</div>
<form id="eval-form">
{"".join(cards_html)}
<button type="button" onclick="downloadAnnotations()">{lbl['submit']}</button>
</form>

<script>
const ITEMS = {items_json};

// Track completion
document.querySelectorAll('input[type=radio]').forEach(r => {{
  r.addEventListener('change', updateProgress);
}});

function updateProgress() {{
  let done = 0;
  for (let i = 1; i <= {len(items)}; i++) {{
    const f = document.querySelector(`input[name=faith_${{i}}]:checked`);
    const a = document.querySelector(`input[name=adeq_${{i}}]:checked`);
    if (f && a) done++;
  }}
  document.getElementById('done').textContent = done;
}}

function downloadAnnotations() {{
  const results = [];
  for (let i = 1; i <= {len(items)}; i++) {{
    const f = document.querySelector(`input[name=faith_${{i}}]:checked`);
    const a = document.querySelector(`input[name=adeq_${{i}}]:checked`);
    const c = document.querySelector(`textarea[name=comment_${{i}}]`);
    results.push({{
      item_id:       i,
      exp_name:      ITEMS[i-1].exp_name,
      query:         ITEMS[i-1].query,
      answer:        ITEMS[i-1].answer,
      faithfulness:  f ? parseInt(f.value) : null,
      adequacy:      a ? parseInt(a.value) : null,
      comment:       c ? c.value : "",
    }});
  }}
  const blob = new Blob([JSON.stringify(results, null, 2)], {{type: "application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'annotations_{lang}.json';
  a.click();
}}
</script>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    print(f"  Wrote HTML: {path}")


# ── Annotation instructions ───────────────────────────────────────────────────

INSTRUCTIONS = """
ANNOTATOR INSTRUCTIONS — Cross-Lingual RAG Faithfulness Evaluation
===================================================================

Thank you for participating in this evaluation. You will assess 50 question-answer
pairs per language (Arabic and Malay).

For each item you will see:
  1. QUESTION       — The question that was asked
  2. RETRIEVED CONTEXT — Up to 3 passages retrieved from Wikipedia
  3. GENERATED ANSWER — An answer produced by the RAG system

You must rate each item on TWO dimensions:

── FAITHFULNESS (1–5) ─────────────────────────────────────────────
Does the generated answer only contain information from the retrieved context?
  5 = Completely faithful: every claim in the answer is supported by the context
  4 = Largely faithful: very minor paraphrasing or omission
  3 = Mostly faithful: some unsupported claims but the core is grounded
  2 = Partially faithful: significant unsupported claims
  1 = Mostly unfaithful: the answer contradicts or ignores the context
  N/A if the answer is empty or "<extra_id_0>" — rate this 1

── ADEQUACY (1–5) ─────────────────────────────────────────────────
Does the generated answer actually address the question?
  5 = Fully answers the question
  4 = Mostly answers with minor gaps
  3 = Partially answers — some relevant information missing
  2 = Barely addresses the question
  1 = Does not address the question at all

── NOTES ──────────────────────────────────────────────────────────
• You do NOT need to know the correct answer. Focus on what the context says.
• Short answers (1–3 words) can still be faithful if grounded in the context.
• Empty answers and "<extra_id_0>" should receive Faithfulness=1, Adequacy=1.
• Use the comment field to flag anything unusual.

── SUBMITTING ─────────────────────────────────────────────────────
Click "Download Annotations (JSON)" when done.
Email the JSON file to the research team along with your annotator ID.

Inter-annotator agreement will be measured with Krippendorff's alpha.
Each item will be rated by 2 annotators independently.

Thank you.
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write instructions
    instr_path = OUTPUT_DIR / "ANNOTATOR_INSTRUCTIONS.txt"
    instr_path.write_text(INSTRUCTIONS, encoding="utf-8")
    print(f"  Wrote instructions: {instr_path}")

    for lang in ["ar", "ms"]:
        print(f"\nPreparing {lang.upper()} evaluation …")
        items = load_experiment_samples(lang)
        if not items:
            print(f"  ⚠  No samples found for lang={lang}. "
                  "Run experiments first: python run_all_experiments.py")
            continue

        print(f"  Total pool: {len(items)} items")
        sample = stratified_sample(items, SAMPLE_SIZE)
        print(f"  Sampled: {len(sample)} items")

        write_csv(sample, OUTPUT_DIR / f"human_eval_{lang}.csv")
        write_html(sample, OUTPUT_DIR / f"annotation_form_{lang}.html", lang)

    print(f"\n✅ Human eval materials saved to: {OUTPUT_DIR}/")
    print("Next steps:")
    print("  1. Open annotation_form_ar.html and annotation_form_ms.html in a browser")
    print("  2. Send to annotators with ANNOTATOR_INSTRUCTIONS.txt")
    print("  3. Collect JSON responses and run: python evaluation/compute_iaa.py")


if __name__ == "__main__":
    main()
