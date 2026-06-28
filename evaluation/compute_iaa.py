"""
evaluation/compute_iaa.py
=========================
Compute inter-annotator agreement (IAA) from two annotation JSON files
and produce a summary table for the paper.

Metrics:
  - Krippendorff's alpha (ordinal) — standard for Likert-scale annotation
  - Cohen's kappa (per dimension)
  - Pearson correlation

Usage
-----
    python evaluation/compute_iaa.py \\
        --ann1 human_eval/annotations_ar_annotator1.json \\
        --ann2 human_eval/annotations_ar_annotator2.json \\
        --output human_eval/iaa_report_ar.txt
"""

import argparse
import json
import math
from pathlib import Path


# ── Krippendorff's alpha (ordinal) ──────────────────────────────────────────

def krippendorff_alpha_ordinal(ratings: list) -> float:
    """
    Compute Krippendorff's alpha for ordinal data.
    ratings: list of (rater1_score, rater2_score) tuples
             scores are integers 1–5; None = missing
    Returns alpha in [-1, 1]; alpha >= 0.8 = reliable, >= 0.67 = acceptable.
    """
    valid = [(a, b) for a, b in ratings if a is not None and b is not None]
    if len(valid) < 2:
        return float("nan")

    n = len(valid)
    # Ordinal distance function: d(a,b)^2 for ordinal scale
    def d2(a, b):
        return (a - b) ** 2

    # Observed disagreement
    Do = sum(d2(a, b) for a, b in valid) / n

    # Expected disagreement (under random agreement)
    all_vals = [v for pair in valid for v in pair]
    n_total = len(all_vals)
    De = sum(
        d2(all_vals[i], all_vals[j])
        for i in range(n_total)
        for j in range(i + 1, n_total)
    ) / (n_total * (n_total - 1) / 2)

    if De == 0:
        return 1.0
    return 1.0 - Do / De


def cohen_kappa(ratings: list) -> float:
    """Simple Cohen's kappa for two raters, integer scores 1–5."""
    valid = [(a, b) for a, b in ratings if a is not None and b is not None]
    if not valid:
        return float("nan")

    n = len(valid)
    agree = sum(1 for a, b in valid if a == b)
    po = agree / n

    from collections import Counter
    r1 = Counter(a for a, _ in valid)
    r2 = Counter(b for _, b in valid)
    pe = sum((r1[k] / n) * (r2[k] / n) for k in set(r1) | set(r2))

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def pearson_r(xs: list, ys: list) -> float:
    valid = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(valid) < 2:
        return float("nan")
    n = len(valid)
    mx = sum(x for x, _ in valid) / n
    my = sum(y for _, y in valid) / n
    num = sum((x - mx) * (y - my) for x, y in valid)
    den = math.sqrt(
        sum((x - mx) ** 2 for x, _ in valid) *
        sum((y - my) ** 2 for _, y in valid)
    )
    return num / den if den > 0 else float("nan")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann1",   required=True, help="Annotation JSON from annotator 1")
    p.add_argument("--ann2",   required=True, help="Annotation JSON from annotator 2")
    p.add_argument("--output", default=None,  help="Save report to this text file")
    args = p.parse_args()

    def load(path):
        with open(path, encoding="utf-8") as f:
            return {item["item_id"]: item for item in json.load(f)}

    ann1 = load(args.ann1)
    ann2 = load(args.ann2)

    common_ids = sorted(set(ann1) & set(ann2))
    if not common_ids:
        print("No common item IDs found between the two annotation files.")
        return

    faith_pairs = [(ann1[i].get("faithfulness"), ann2[i].get("faithfulness")) for i in common_ids]
    adeq_pairs  = [(ann1[i].get("adequacy"),     ann2[i].get("adequacy"))     for i in common_ids]

    faith_alpha = krippendorff_alpha_ordinal(faith_pairs)
    faith_kappa = cohen_kappa(faith_pairs)
    faith_r     = pearson_r(
        [a for a, _ in faith_pairs if a is not None],
        [b for _, b in faith_pairs if b is not None],
    )

    adeq_alpha = krippendorff_alpha_ordinal(adeq_pairs)
    adeq_kappa = cohen_kappa(adeq_pairs)
    adeq_r     = pearson_r(
        [a for a, _ in adeq_pairs if a is not None],
        [b for _, b in adeq_pairs if b is not None],
    )

    report = f"""
Inter-Annotator Agreement Report
==================================
Items compared: {len(common_ids)}

FAITHFULNESS
  Krippendorff's alpha (ordinal): {faith_alpha:.4f}
  Cohen's kappa:                  {faith_kappa:.4f}
  Pearson r:                      {faith_r:.4f}

ADEQUACY
  Krippendorff's alpha (ordinal): {adeq_alpha:.4f}
  Cohen's kappa:                  {adeq_kappa:.4f}
  Pearson r:                      {adeq_r:.4f}

Interpretation (Krippendorff's alpha):
  alpha >= 0.80 = reliable
  alpha >= 0.67 = acceptable (tentative conclusions)
  alpha <  0.67 = unreliable — requires adjudication

Note: for paper reporting, use Krippendorff's alpha (ordinal) as the primary metric.
"""

    print(report)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
