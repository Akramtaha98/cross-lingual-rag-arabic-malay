# Cross-Lingual RAG: Arabic ↔ Malay

**Diagnosing retrieval and evaluation failures in cross-lingual Retrieval-Augmented Generation for two typologically distant, resource-asymmetric languages (Arabic and Malay).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-under%20review-yellow)]()

---

## What this is

A diagnostic benchmark for cross-lingual RAG on **Arabic (AR)** and **Malay (MS)** — a Semitic, non-concatenative, RTL language paired with an Austronesian, agglutinative, low-resource one. Five retrievers, five controlled experiments, bootstrap-validated confidence intervals, and an error analysis that isolates *where* cross-lingual RAG actually fails: retrieval, not generation.

**Key findings:**
- Retrieval is a binding but not the only bottleneck: Recall@5 is below 2% for every retriever, and the gold answer appears anywhere in the 50K-passage corpus for only about half of the questions (answer coverage).
- mE5-large is the strongest retriever; Arabic normalization gives no gain once the relevance check is held fixed.
- Under *matched* retrieval (mE5-large, n=200), Qwen2.5-7B's BERTScore-F advantage over mT5+LoRA shrinks from about 14% to under 3%, and LLM-judged adequacy stays near floor for all generators.
- Supplying the gold passage (oracle context) does not improve mT5 output, so answer extraction is also limiting.
- No generator produces correct answers: exact match is 0 for every mT5 and Qwen configuration (BERTScore-F is moderate but does not measure correctness).
- The Arabic→Malay vs. Malay→Arabic transfer asymmetry is not statistically significant (paired bootstrap, p≈0.13).

---

## Highlights

- 🔍 **5 retrievers** — BM25, BM25+Norm, BM25+CAMeL (morphological stemming), LaBSE, mE5-base/large, BGE-M3
- 🌐 **Bidirectional cross-lingual retrieval** — AR→MS and MS→AR, isolating the transfer asymmetry
- 🧠 **Generation study** — mT5-small base vs. LoRA-adapted vs. Qwen2.5-7B-Instruct, with a strictly held-out split (MKQA items 1–1,000 for evaluation, 1,001–10,000 for LoRA training; zero overlap verified)
- 📊 **Bootstrap 95% CIs** on every retrieval metric (1,000 resamples, cross-checked against normal approximation)
- 🤖 **LLM-as-judge evaluation** (Faithfulness + Adequacy) with a local Qwen2.5-7B judge (Ollama) and claude-haiku-4-5, alongside lexical metrics
- 🧪 **Oracle-context, answer-coverage and matched-retriever experiments**, plus paired-bootstrap significance tests
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
├── config.py
├── run_all_experiments.py       # EXP1–EXP5 driver
├── run_normalized_bm25.py       # BM25+Norm (diacritic/alef normalization)
├── run_camel_bm25.py            # BM25+CAMeL (full morphological stemming)
├── run_strong_generator.py      # Qwen2.5-7B-Instruct comparison
├── train_lora.py                # LoRA fine-tuning
├── compute_bootstrap_ci.py      # bootstrap resampling for all metrics
├── answer_coverage.py           # share of gold answers present in the corpus
├── run_oracle_context.py / run_oracle_context_v2.py / run_oracle_gen.py  # oracle & closed-book
├── paired_oracle_sig.py         # paired bootstrap significance tests
├── precompute_matched_retrieval.py + gen_from_cached_retrieval.py  # matched-retriever Table 9
├── judge_ollama.py              # local LLM judge (Ollama, qwen2.5:7b)
├── answer_level_metrics.py      # normalized EM / token F1 / Contains-gold
├── crosslingual_parallel_relevance.py  # cross-lingual retrieval re-scored with corpus-language gold answers
├── run_qwen_oracle.py           # Qwen2.5-7B closed-book vs. oracle context
├── recompute_rouge.py           # ROUGE-L with Unicode-aware tokenization
├── update_paper_tables.py
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Akramtaha98/cross-lingual-rag-arabic-malay.git
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

[MKQA](https://github.com/apple/mkqa) — to our knowledge the only open-domain QA benchmark with parallel Arabic and Malay tracks. Retrieval corpora are built from language-specific Wikipedia snapshots (November 2023), chunked into overlapping 200-word passages and indexed with FAISS.

---

## Experiments

```bash
python3 run_all_experiments.py        # EXP1–EXP5
python3 run_normalized_bm25.py        # BM25+Norm ablation
python3 run_camel_bm25.py             # BM25+CAMeL ablation
python3 run_strong_generator.py       # mT5 vs. Qwen2.5-7B
python3 compute_bootstrap_ci.py       # confidence intervals for all results
python3 answer_coverage.py            # corpus answer coverage
python3 run_oracle_context_v2.py      # oracle / closed-book context
python3 paired_oracle_sig.py          # paired significance tests
```

1. **Retriever comparison** — sparse vs. dense, monolingual, both languages
2. **Monolingual vs. cross-lingual** — retrieval degradation when query and corpus languages differ
3. **LoRA fine-tuning** — does adaptation improve generation quality and faithfulness?
4. **Top-K ablation** — passage count vs. retrieval/generation tradeoff
5. **Transfer direction asymmetry** — AR→MS vs. MS→AR, with a four-factor linguistic account

---

## Evaluation

**Retrieval:** Hit@K (top-K contains ≥1 relevant passage), Recall@K, Precision@K, MRR, NDCG@10 (bootstrap 95% CIs)
**Generation:** BLEU-4, ROUGE-L, BERTScore-F, lexical Faithfulness
**LLM-as-judge:** Faithfulness + Adequacy (1–5 scale), scored with a local `qwen2.5:7b` judge via Ollama (`judge_ollama.py`); no paid API is required

---

## Reproducibility

Everything needed to reproduce every table and figure in the paper is here: source code, FAISS index builders, LoRA training config, evaluation scripts, and raw result JSONs in `results/`. Random seeds are fixed (`seed=42`) and the LoRA train/test split is verified to have zero item overlap.

---

## Citation

```bibtex
@article{najm2026crosslingualrag,
  author={Najm, Sarah Mohammed and Salman, Aymen D. and Al-Dujaili, Ayad and Humaidi, Amjad J.},
  title={Diagnosing Retrieval and Evaluation Failures in Cross-Lingual Retrieval-Augmented Generation for Arabic and Malay},
  journal={Computers (MDPI)},
  year={2026},
  note={Under review}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## Large files

FAISS indexes (`indexes/`, ~2 GB), model weights and the fastText `lid.176.bin` are not tracked. Rebuild the indexes with the scripts above; `lid.176.bin` is downloadable from fastText.

---

## Acknowledgements

Built on [MKQA](https://github.com/apple/mkqa), [Hugging Face Transformers](https://github.com/huggingface/transformers), [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers), [FAISS](https://github.com/facebookresearch/faiss), [PEFT](https://github.com/huggingface/peft), [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools), [rank-bm25](https://github.com/dorianbrown/rank_bm25), and [SacreBLEU](https://github.com/mjpost/sacrebleu).

## Latest results (corrected)
- Cross-lingual retrieval is rescored with corpus-language gold answers (`crosslingual_parallel_relevance.py`, results in `results/xl_parallel_*.json`). ar→ms and ms→ar are tied (mE5-large MRR 0.059 vs 0.059; LaBSE 0.027 vs 0.027; paired bootstrap p ≥ 0.93).
- Qwen2.5-7B closed-book vs oracle (`run_qwen_oracle_hf.py`, `results/qwen_oracle_*.json`, n=100/lang): the answer-string "oracle" passage does not help and is often off-topic, so it does not isolate retrieval from generation.
- Answer-level metrics (EM / F1 / Contains): `answer_level_metrics.py`.
