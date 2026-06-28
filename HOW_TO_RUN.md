# How to Run the Cross-Lingual RAG Experiments

## 1. Install dependencies
```bash
cd cross_lingual_rag
pip install -r requirements.txt
```

## 2. Set your LoRA checkpoint path
Edit `config.py`:
```python
LORA_MODEL_PATH = "./models/mt5_lora"   # ← your fine-tuned checkpoint folder
```
The folder should contain: `adapter_config.json` + `adapter_model.safetensors` (or `.bin`).

## 3. Quick dev run (fast, 500 samples)
```bash
python run_all_experiments.py --dev
```

## 4. Run a specific experiment only
```bash
python run_all_experiments.py --exp exp1    # EXP 1: retriever comparison
python run_all_experiments.py --exp exp2    # EXP 2: language direction
python run_all_experiments.py --exp exp3    # EXP 3: generator tuning
python run_all_experiments.py --exp exp4    # EXP 4: top-K ablation
python run_all_experiments.py --exp exp5    # EXP 5: transfer direction gap
```

## 5. Full run (paper-quality, ~10k samples)
```bash
python run_all_experiments.py
```

## 6. Force rebuild FAISS indexes
```bash
python run_all_experiments.py --rebuild_index
```

## Output
All results saved to `./results/`:
- `<exp_name>_results.json` — per-experiment metrics + 10 sample outputs
- `summary_table.csv` — all experiments in one table (copy to your paper)

## Project structure
```
cross_lingual_rag/
├── config.py                  ← all settings + experiment matrix
├── requirements.txt
├── run_all_experiments.py     ← MAIN ENTRY POINT
├── data/
│   ├── load_mkqa.py           ← MKQA dataset loader (Arabic + Malay)
│   └── load_wikipedia.py      ← Wikipedia corpus builder + chunker
├── retrieval/
│   ├── bm25_retriever.py      ← BM25 sparse baseline
│   ├── dense_retriever.py     ← LaBSE + mE5 dense retrievers
│   └── faiss_index.py         ← FAISS index (build, save, load)
├── generation/
│   └── mt5_generator.py       ← mT5-small + LoRA generator
├── pipeline/
│   └── rag_pipeline.py        ← full RAG orchestration
└── evaluation/
    └── metrics.py             ← all 8 metrics (Recall@K, MRR, NDCG, BLEU, ROUGE-L, BERTScore, Faithfulness)
```

## What each experiment tests

| Exp | Variable | Configs |
|-----|----------|---------|
| EXP 1 | Retriever type | BM25 vs. LaBSE vs. mE5 (both languages) |
| EXP 2 | Language direction | Monolingual vs. cross-lingual retrieval |
| EXP 3 | Generator tuning | mT5 base vs. mT5 + LoRA |
| EXP 4 | Top-K | K=3 vs. K=5 vs. K=10 |
| EXP 5 | Transfer direction | AR→MS gap vs. MS→AR gap |

## Tips
- First time: run `--dev` to verify everything works (takes ~15 min on CPU)
- FAISS indexes are cached to `./indexes/` — only built once per retriever/corpus combo
- Wikipedia corpora are cached to `./indexes/corpus_ar.json` and `corpus_ms.json`
- GPU recommended for generation; retrieval runs fine on CPU
