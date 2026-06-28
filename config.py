"""
config.py — Central configuration for the Cross-Lingual RAG experiment.
Edit LORA_MODEL_PATH and OUTPUT_DIR before running.
"""

from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
LORA_MODEL_PATH = "./models/mt5_lora"   # ← point to your fine-tuned checkpoint
BASE_MODEL_NAME  = "google/mt5-small"
OUTPUT_DIR       = "./results"
INDEX_DIR        = "./indexes"

# ──────────────────────────────────────────────
# Languages
# ──────────────────────────────────────────────
LANGUAGES = {
    "ar": "Arabic",
    "ms": "Malay",
}

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
MKQA_LANGUAGES = ["ar", "ms"]          # both present in MKQA
MKQA_MAX_SAMPLES = 1000                # use full 10k for final paper; 1k for dev

# Wikipedia corpus settings
WIKI_CHUNK_SIZE     = 200              # words per passage
WIKI_CHUNK_OVERLAP  = 50              # overlapping words between chunks
WIKI_MAX_PASSAGES   = 50_000          # cap per language (RAM/time tradeoff)

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
RETRIEVER_MODELS = {
    "labse":     "sentence-transformers/LaBSE",
    "me5":       "intfloat/multilingual-e5-base",
    # Stronger baselines (required for Computational Linguistics / IPM level submission)
    "me5large":  "intfloat/multilingual-e5-large",   # ~560M params vs 278M for base
    "bgem3":     "BAAI/bge-m3",                       # hybrid dense+sparse, SOTA multilingual
}
TOP_K_VALUES = [3, 5, 10]              # EXP 4: ablation over K
DEFAULT_TOP_K = 5

# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────
MAX_INPUT_LENGTH  = 512
MAX_TARGET_LENGTH = 128
GENERATION_KWARGS = {
    "max_new_tokens": 128,
    "num_beams": 4,
    "early_stopping": True,
}

# ──────────────────────────────────────────────
# Experiment matrix (EXP 1–5)
# ──────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    name:         str
    retriever:    str          # "bm25" | "labse" | "me5" | "me5large" | "bgem3"
    query_lang:   str          # "ar" | "ms"
    corpus_lang:  str          # "ar" | "ms" | "both"
    top_k:        int
    use_lora:     bool         # True = your fine-tuned model; False = base mT5


EXPERIMENTS: List[ExperimentConfig] = [
    # ── EXP 1: Retriever comparison (Arabic monolingual) ──
    ExperimentConfig("exp1_bm25_ar",      "bm25",    "ar", "ar", 5, True),
    ExperimentConfig("exp1_labse_ar",     "labse",   "ar", "ar", 5, True),
    ExperimentConfig("exp1_me5_ar",       "me5",     "ar", "ar", 5, True),
    # Stronger baselines — bgem3 and me5large forced to CPU (MPS OOM fixed in dense_retriever.py)
    # Run separately (each takes ~20-30 min on CPU for 50k passages):
    #   python run_all_experiments.py --exp exp1_bgem3 --skip-existing
    #   python run_all_experiments.py --exp exp1_me5large --skip-existing
    ExperimentConfig("exp1_me5large_ar",  "me5large","ar", "ar", 5, True),
    ExperimentConfig("exp1_bgem3_ar",     "bgem3",   "ar", "ar", 5, True),
    # Malay monolingual
    ExperimentConfig("exp1_bm25_ms",      "bm25",    "ms", "ms", 5, True),
    ExperimentConfig("exp1_labse_ms",     "labse",   "ms", "ms", 5, True),
    ExperimentConfig("exp1_me5_ms",       "me5",     "ms", "ms", 5, True),
    ExperimentConfig("exp1_me5large_ms",  "me5large","ms", "ms", 5, True),
    ExperimentConfig("exp1_bgem3_ms",     "bgem3",   "ms", "ms", 5, True),

    # ── EXP 2: Language direction (mono vs. cross) ──
    ExperimentConfig("exp2_ar_mono",   "labse", "ar", "ar",   5, True),
    ExperimentConfig("exp2_ar_cross",  "labse", "ar", "ms",   5, True),
    ExperimentConfig("exp2_ms_mono",   "labse", "ms", "ms",   5, True),
    ExperimentConfig("exp2_ms_cross",  "labse", "ms", "ar",   5, True),

    # ── EXP 3: Generator tuning (base vs. LoRA) ──
    ExperimentConfig("exp3_base_ar",   "labse", "ar", "ar", 5, False),
    ExperimentConfig("exp3_lora_ar",   "labse", "ar", "ar", 5, True),
    ExperimentConfig("exp3_base_ms",   "labse", "ms", "ms", 5, False),
    ExperimentConfig("exp3_lora_ms",   "labse", "ms", "ms", 5, True),

    # ── EXP 4: Top-K ablation ──
    ExperimentConfig("exp4_k3",        "labse", "ar", "ar", 3,  True),
    ExperimentConfig("exp4_k5",        "labse", "ar", "ar", 5,  True),
    ExperimentConfig("exp4_k10",       "labse", "ar", "ar", 10, True),

    # ── EXP 5: Transfer direction gap ──
    ExperimentConfig("exp5_ar_to_ms",  "labse", "ar", "ms", 5, True),
    ExperimentConfig("exp5_ms_to_ar",  "labse", "ms", "ar", 5, True),
]
