"""
train_lora.py — Fine-tune mT5-small on MKQA (ar + ms) with LoRA.

Saves the adapter to LORA_MODEL_PATH (./models/mt5_lora) so that
run_all_experiments.py can load it with use_lora=True.

Usage:
    python3 train_lora.py               # full training
    python3 train_lora.py --dev         # quick smoke-test (100 samples, 1 epoch)
"""

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BASE_MODEL_NAME,
    LORA_MODEL_PATH,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    MKQA_LANGUAGES,
)
from data.load_mkqa import load_mkqa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ── LoRA hyper-parameters ────────────────────────────────────────────────────
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.1
TARGET_MODULES  = ["q", "v"]          # mT5 attention projections

# ── Training hyper-parameters ────────────────────────────────────────────────
NUM_EPOCHS      = 3
BATCH_SIZE      = 8
LEARNING_RATE   = 3e-4
WARMUP_STEPS    = 100
GRAD_ACCUM      = 4                   # effective batch = BATCH_SIZE * GRAD_ACCUM


# ─────────────────────────────────────────────────────────────────────────────
class MKQADataset(Dataset):
    """Flat list of (input_str, target_str) pairs from MKQA ar + ms splits."""

    def __init__(self, pairs):
        self.pairs = pairs          # list of (question, answer) strings

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def build_pairs(max_samples: int = None) -> list:
    """Load MKQA for all languages and return (input_text, answer) pairs."""
    # load_mkqa returns {"ar": [{"query": ..., "answer": ...}, ...], "ms": [...]}
    data = load_mkqa(languages=MKQA_LANGUAGES, max_samples=max_samples)
    pairs = []
    for lang in MKQA_LANGUAGES:
        for item in data.get(lang, []):
            question    = item.get("query", "").strip()
            answer_text = item.get("answer", "").strip()
            if question and answer_text:
                input_text = f"question: {question} context:"
                pairs.append((input_text, answer_text))
    logger.info(f"Total training pairs: {len(pairs)}")
    return pairs


def collate_fn(tokenizer, batch):
    inputs, targets = zip(*batch)
    model_inputs = tokenizer(
        list(inputs),
        max_length=MAX_INPUT_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            list(targets),
            max_length=MAX_TARGET_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).input_ids
    # Replace padding token id with -100 so loss ignores padding
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def train(dev_mode: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    max_samples = 100 if dev_mode else None
    pairs = build_pairs(max_samples=max_samples)
    if not pairs:
        raise RuntimeError(
            "No training pairs found. "
            "Make sure MKQA data is accessible (run_all_experiments.py downloads it)."
        )

    dataset  = MKQADataset(pairs)
    num_epochs = 1 if dev_mode else NUM_EPOCHS

    # ── Tokenizer ────────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(tokenizer, b),
    )

    # ── Model + LoRA ─────────────────────────────────────────────────────────
    logger.info(f"Loading base model: {BASE_MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)
    model.train()

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    total_steps = (len(loader) // GRAD_ACCUM) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(WARMUP_STEPS, total_steps // 10),
        num_training_steps=total_steps,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    logger.info(f"Training for {num_epochs} epoch(s), {len(dataset)} examples …")
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                logger.info(f"  epoch {epoch+1}  step {step+1}/{len(loader)}  loss={outputs.loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} — avg loss: {avg_loss:.4f}")

    # ── Save adapter ─────────────────────────────────────────────────────────
    os.makedirs(LORA_MODEL_PATH, exist_ok=True)
    model.save_pretrained(LORA_MODEL_PATH)
    tokenizer.save_pretrained(LORA_MODEL_PATH)
    logger.info(f"LoRA adapter saved to: {LORA_MODEL_PATH}")
    logger.info("Done. You can now run: python3 run_all_experiments.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Quick smoke-test: 100 samples, 1 epoch")
    args = parser.parse_args()
    train(dev_mode=args.dev)
