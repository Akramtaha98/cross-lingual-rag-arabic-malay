# Cross-Lingual Retrieval-Augmented Generation for Arabic and Malay

> **Official implementation** accompanying the research paper:

**Cross-Lingual Retrieval-Augmented Generation for Arabic and Malay:
Dense vs. Sparse Retrieval with Parameter-Efficient Fine-Tuning**

------------------------------------------------------------------------

## Overview

This repository provides a complete implementation of a **Cross-Lingual
Retrieval-Augmented Generation (RAG)** framework for **Arabic (AR)** and
**Malay (MS)** question answering.

The project investigates retrieval and generation across typologically
distinct languages using multilingual dense retrieval, sparse retrieval,
and parameter-efficient fine-tuning.

### Highlights

-   End-to-end Cross-Lingual RAG pipeline
-   Arabic ↔ Malay retrieval and generation
-   BM25, LaBSE, mE5-base, mE5-large, and BGE-M3 evaluation
-   mT5-small with LoRA fine-tuning (PEFT)
-   MKQA benchmark evaluation
-   FAISS indexing
-   Retrieval, generation, and LLM-as-Judge evaluation
-   Reproducible experimental pipeline

------------------------------------------------------------------------

## Repository Structure

``` text
cross_lingual_rag/
├── data/
├── retrieval/
├── pipeline/
├── evaluation/
├── indexes/
├── models/
├── results/
├── figures/
├── scripts/
├── config.py
├── run_all_experiments.py
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## Installation

``` bash
git clone https://github.com/Akramtaha98/cross-lingual-rag-arabic-malay.git
cd cross-lingual-rag-arabic-malay

python -m venv .venv
source .venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
```

------------------------------------------------------------------------

## Models

### Retrieval

  Model                   Type
  ----------------------- ---------------------------------
  BM25                    Sparse Retrieval
  LaBSE                   Dense Multilingual Retrieval
  multilingual-e5-base    Dense Retrieval
  multilingual-e5-large   Dense Retrieval
  BGE-M3                  Hybrid Dense + Sparse Retrieval

### Generator

-   mT5-small
-   mT5-small + LoRA (PEFT)

------------------------------------------------------------------------

## Dataset

Experiments are conducted on the **MKQA** multilingual question
answering benchmark using Arabic and Malay.

Language-specific Wikipedia corpora are indexed using FAISS for
efficient retrieval.

------------------------------------------------------------------------

## Experiments

The repository reproduces the following experiments:

1.  Retriever comparison
2.  Monolingual vs. Cross-lingual retrieval
3.  LoRA fine-tuning
4.  Top-K retrieval ablation
5.  Arabic ↔ Malay transfer asymmetry

------------------------------------------------------------------------

## Evaluation Metrics

### Retrieval

-   Recall@K
-   Precision@K
-   Mean Reciprocal Rank (MRR)
-   NDCG@10

### Generation

-   Exact Match (EM)
-   Token F1
-   BLEU-4
-   ROUGE-L
-   BERTScore
-   Faithfulness

### LLM-based Evaluation

-   Faithfulness
-   Adequacy

------------------------------------------------------------------------

## Reproducibility

The repository includes:

-   Complete source code
-   Configuration files
-   Experiment scripts
-   Evaluation pipeline
-   FAISS index generation
-   LoRA training scripts
-   Result processing utilities

------------------------------------------------------------------------

## Citation

``` bibtex
@article{taha2026crosslingualrag,
  title={Cross-Lingual Retrieval-Augmented Generation for Arabic and Malay: Dense vs. Sparse Retrieval with Parameter-Efficient Fine-Tuning},
  author={Akram Taha and Fanan Hikmat},
  journal={Under Review},
  year={2026}
}
```

------------------------------------------------------------------------

## License

This project is released under the MIT License.

------------------------------------------------------------------------

## Contact

**Akram Taha**

PhD Candidate\
Faculty of Information Science and Technology\
Universiti Kebangsaan Malaysia (UKM)

Email: akramtaha20@gmail.com

GitHub: https://github.com/Akramtaha98

------------------------------------------------------------------------

## Acknowledgements

This work builds upon:

-   MKQA
-   Hugging Face Transformers
-   Sentence Transformers
-   FAISS
-   PEFT
-   PyTorch
-   SacreBLEU
-   BERTScore
-   rank-bm25

If you find this repository useful, please consider citing our paper.
