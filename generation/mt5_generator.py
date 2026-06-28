"""
generation/mt5_generator.py
mT5-small generator with optional LoRA fine-tuning.

Supports:
  - Base mT5-small (no fine-tuning) — EXP 3 baseline
  - mT5-small + your LoRA checkpoint — EXP 3 proposed

The RAG input format:
  "question: <query> context: <doc1> <doc2> <doc3>"
mT5 generates the answer autoregressively.

BUG FIX (v2): mT5's default decoder_start_token_id is the pad token (id=0),
but without explicitly setting it some transformers versions start decoding
from <extra_id_0> (the T5 span-filling sentinel).  We now:
  1. Force decoder_start_token_id = tokenizer.pad_token_id in generate()
  2. Post-process outputs to strip any remaining <extra_id_X> tokens
     (regex covers <extra_id_0> through <extra_id_99>)
"""

import logging
import re
import torch
from typing import List, Optional, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from config import (
    LORA_MODEL_PATH,
    BASE_MODEL_NAME,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    GENERATION_KWARGS,
)

# Regex to strip T5 span-filling sentinel tokens from decoded text
_EXTRA_ID_RE = re.compile(r"<extra_id_\d+>\s*")

logger = logging.getLogger(__name__)


def build_input(query: str, passages: List[Dict], max_passages: int = 5) -> str:
    """
    Format RAG input for mT5:
      "question: <query> context: <passage1_text> [SEP] <passage2_text> ..."

    Truncates to top max_passages to keep within token limits.
    """
    context_parts = [p["text"] for p, _ in passages[:max_passages]]
    context = " [SEP] ".join(context_parts)
    return f"question: {query} context: {context}"


class MT5Generator:
    """
    Seq2Seq generator based on mT5-small.
    Set use_lora=True to load your LoRA adapter from LORA_MODEL_PATH.
    """

    def __init__(self, use_lora: bool = True, device: Optional[str] = None):
        self.use_lora = use_lora
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        logger.info(f"Loading tokenizer: {BASE_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        logger.info(f"Loading base model: {BASE_MODEL_NAME}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

        if use_lora:
            logger.info(f"Loading LoRA adapter from: {LORA_MODEL_PATH}")
            self.model = PeftModel.from_pretrained(self.model, LORA_MODEL_PATH)
            self.model = self.model.merge_and_unload()  # merge for faster inference
            logger.info("LoRA merged into base model weights.")

        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        query: str,
        retrieved_passages: List[Dict],
        top_k: int = 5,
        **kwargs,
    ) -> str:
        """
        Generate answer for a single query given retrieved passages.

        Args:
            query: question string
            retrieved_passages: list of (passage_dict, score) tuples
            top_k: number of passages to include in context

        Returns:
            Generated answer string
        """
        input_text = build_input(query, retrieved_passages, max_passages=top_k)

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
        ).to(self.device)

        gen_kwargs = {
            **GENERATION_KWARGS,
            # Explicitly set decoder start token to pad (not <extra_id_0>).
            # This prevents mT5 from entering span-filling mode.
            "decoder_start_token_id": self.tokenizer.pad_token_id,
            **kwargs,
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        # decode with skip_special_tokens; then strip any residual <extra_id_X>
        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = _EXTRA_ID_RE.sub("", raw).strip()
        return answer

    def batch_generate(
        self,
        queries: List[str],
        retrieved_passages_list: List[List[Dict]],
        top_k: int = 5,
        batch_size: int = 8,
    ) -> List[str]:
        """
        Generate answers for a batch of queries.
        retrieved_passages_list[i] corresponds to queries[i].
        """
        answers = []
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]
            batch_passages = retrieved_passages_list[i : i + batch_size]

            input_texts = [
                build_input(q, p, max_passages=top_k)
                for q, p in zip(batch_queries, batch_passages)
            ]

            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding=True,
            ).to(self.device)

            batch_gen_kwargs = {
                **GENERATION_KWARGS,
                "decoder_start_token_id": self.tokenizer.pad_token_id,
            }
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **batch_gen_kwargs,
                )

            for out in outputs:
                raw = self.tokenizer.decode(out, skip_special_tokens=True)
                answers.append(_EXTRA_ID_RE.sub("", raw).strip())
            logger.info(f"Generated {min(i + batch_size, len(queries))}/{len(queries)}")

        return answers
