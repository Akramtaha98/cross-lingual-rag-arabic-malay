# Cross-Lingual RAG: Arabic ↔ Malay

**Diagnosing the retrieval bottleneck in cross-lingual Retrieval-Augmented Generation for two typologically distant, resource-asymmetric languages.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-under%20review-yellow)]()

---

## What this is

A diagnostic benchmark for cross-lingual RAG on **Arabic (AR)** and **Malay (MS)** — a Semitic, non-concatenative, RTL language paired with an Austronesian, agglutinative, low-resource one. Five retrievers, five controlled experiments, bootstrap-validated confidence intervals, and an error analysis that isolates *where* cross-lingual RAG actually fails: retrieval, not generation.

**Key finding:** even a 23×-larger generator (Qwen2.5-7B vs. mT5-small) doesn't move the needle on answer adequacy, because the retrieved context is almost never relevant in the first place. Better generators can't fix a retrieval problem.

---

## Highlights

- 🔍 **5 retrievers** — BM25, BM25+Norm, BM25+CAMeL (morphological stemming), LaBSE, mE5-base/large, BGE-M3
- 🌐 **Bidirectional cross-lingual retrieval** — AR→MS and MS→AR, isolating the transfer asymmetry
- 🧠 **Generation study** — mT5-small base vs. LoRA-adapted vs. Qwen2.5-7B-Instruct, with a strictly held-out train/test split (zero overlap verified)
- 📊 **Bootstrap 95% CIs** on every retrieval metric (1,000 resamples, cross-checked against normal approximation)
- 🤖 **LLM-as-judge evaluation** (Faithfulness + Adequacy) alongside lexical metrics
- 🗣️ **Measured code-switching** — fastText language-ID pass over the full 50K-passage Malay corpus (not just a theoretical claim)
- 📈 **Answer-type breakdown** and manual error analysis on retrieval failures

---

## Repository structure

```
cross_lingual_rag/
├── data/                    # MKQA + Wikipedia corpus loaders
├── retrieval/                # BM25 / dense / FAISS retrievers
├── generation/                # mT5 generator + LoRA
├── pipeline/                  # end-to-end RAG pipeline
├── evaluation/                 # metrics, LLM-judge, human-eval prep
├── human_eval/                 # annotation forms + instructions
├── results/                    # experiment outputs (JSON/CSV)
├── figures/                    # generated plots
├── config.py
├── run_all_experiments.py       # EXP1–EXP5 driver
├── run_normalized_bm25.py       # BM25+Norm (diacritic/alef normalization)
├── run_camel_bm25.py            # BM25+CAMeL (full morphological stemming)
├── run_strong_generator.py      # Qwen2.5-7B-Instruct comparison
├── train_lora.py                # LoRA fine-tuning
├── compute_bootstrap_ci.py      # bootstrap resampling for all metrics
├── update_paper_tables.py
└── requirements.txt
```

---

## Setup

```bash
git clone <this-repository-url>
cd cross-lingual-rag-arabic-malay

python -m venv .venv
source .venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
```

---

## Retrievers

| Model | Type |
|---|---|
| BM25 | Sparse |
| BM25+Norm | Sparse + Arabic diacritic/alef normalization |
| BM25+CAMeL | Sparse + full CAMeL Tools morphological stemming |
| LaBSE | Dense, multilingual bi-encoder |
| multilingual-e5-base / -large | Dense |
| BGE-M3 | Hybrid dense + sparse |

## Generators

- mT5-small (baseline)
- mT5-small + LoRA (rank 8, held-out train/test split)
- Qwen2.5-7B-Instruct (RAG comparison, 23× larger)

---

## Dataset

[MKQA](https://github.com/apple/mkqa) — the only existing multilingual QA benchmark with parallel Arabic and Malay tracks. Retrieval corpora are built from language-specific Wikipedia snapshots (November 2023), chunked into overlapping 200-word passages and indexed with FAISS.

---

## Experiments

```bash
python3 run_all_experiments.py        # EXP1–EXP5
python3 run_normalized_bm25.py        # BM25+Norm ablation
python3 run_camel_bm25.py             # BM25+CAMeL ablation
python3 run_strong_generator.py       # mT5 vs. Qwen2.5-7B
python3 compute_bootstrap_ci.py       # confidence intervals for all results
```

1. **Retriever comparison** — sparse vs. dense, monolingual, both languages
2. **Monolingual vs. cross-lingual** — retrieval degradation when query and corpus languages differ
3. **LoRA fine-tuning** — does adaptation improve generation quality and faithfulness?
4. **Top-K ablation** — passage count vs. retrieval/generation tradeoff
5. **Transfer direction asymmetry** — AR→MS vs. MS→AR, with a four-factor linguistic account

---

## Evaluation

**Retrieval:** Recall@K, Precision@K, MRR, NDCG@10 (bootstrap 95% CIs)
**Generation:** BLEU-4, ROUGE-L, BERTScore-F, lexical Faithfulness
**LLM-as-judge:** Faithfulness + Adequacy (1–5 scale), scored via `claude-haiku-4-5`

---

## Reproducibility

Everything needed to reproduce every table and figure in the paper is here: source code, FAISS index builders, LoRA training config, evaluation scripts, and raw result JSONs in `results/`. Random seeds are fixed (`seed=42`) and the LoRA train/test split is verified to have zero item overlap.

---

## Citation

```bibtex
@article{crosslingualrag2026,
  title={Diagnosing the Retrieval Bottleneck in Cross-Lingual Retrieval-Augmented Generation for Arabic and Malay},
  journal={Information (MDPI)},
  year={2026},
  note={Under review}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

Built on [MKQA](https://github.com/apple/mkqa), [Hugging Face Transformers](https://github.com/huggingface/transformers), [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers), [FAISS](https://github.com/facebookresearch/faiss), [PEFT](https://github.com/huggingface/peft), [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools), [rank-bm25](https://github.com/dorianbrown/rank_bm25), and [SacreBLEU](https://github.com/mjpost/sacrebleu).
