"""
data/load_mkqa.py
Load MKQA QA pairs for Arabic (ar) and Malay (ms).

MKQA: Longpre et al. (2021) — 10,000 queries × 26 languages.
Source: Apple GitHub (direct JSONL download — avoids deprecated loading script).
URL: https://github.com/apple/ml-mkqa/raw/main/mkqa.jsonl.gz
"""

import os
import gzip
import json
import ssl
import urllib.request
import logging
from typing import List, Dict, Optional
from config import MKQA_LANGUAGES, MKQA_MAX_SAMPLES, INDEX_DIR

logger = logging.getLogger(__name__)

MKQA_URLS = [
    "https://raw.githubusercontent.com/apple/ml-mkqa/main/mkqa.jsonl.gz",
    "https://raw.githubusercontent.com/apple/ml-mkqa/master/mkqa.jsonl.gz",
]
MKQA_CACHE = os.path.join(INDEX_DIR, "mkqa.jsonl.gz")


def _download_mkqa(cache_path: str = MKQA_CACHE) -> str:
    """Download MKQA JSONL from Apple GitHub if not already cached."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.exists(cache_path):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        for url in MKQA_URLS:
            try:
                logger.info(f"Downloading MKQA from {url} …")
                with urllib.request.urlopen(url, context=ctx) as response:
                    with open(cache_path, "wb") as f:
                        f.write(response.read())
                logger.info(f"Saved to {cache_path}")
                return cache_path
            except Exception as e:
                logger.warning(f"Failed ({url}): {e}")

        raise RuntimeError(
            "Could not download MKQA from any source.\n"
            "Please download manually:\n"
            "  1. Go to https://github.com/apple/ml-mkqa\n"
            "  2. Download mkqa.jsonl.gz\n"
            f"  3. Place it at: {cache_path}"
        )
    else:
        logger.info(f"Using cached MKQA: {cache_path}")
    return cache_path


def _iter_mkqa(cache_path: str):
    """Iterate over MKQA rows from the gzipped JSONL file."""
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_mkqa(
    languages: List[str] = MKQA_LANGUAGES,
    max_samples: Optional[int] = MKQA_MAX_SAMPLES,
    answer_type_filter: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """
    Load MKQA for the specified languages.

    Returns:
        {
          "ar": [{"query": "...", "answer": "...", "query_id": "...", "type": "..."}, ...],
          "ms": [...],
        }
    """
    cache_path = _download_mkqa()

    result: Dict[str, List[Dict]] = {lang: [] for lang in languages}
    total = 0

    for row in _iter_mkqa(cache_path):
        if max_samples and total >= max_samples:
            break

        for lang in languages:
            query  = row.get("queries", {}).get(lang)
            answer = row.get("answers", {}).get(lang)

            if not query:
                continue

            # answers is a list of dicts: [{"text": "...", "aliases": [...]}]
            ans_text = ""
            if answer and len(answer) > 0:
                ans_text = answer[0].get("text", "") or ""

            if answer_type_filter and row.get("type") != answer_type_filter:
                continue

            result[lang].append({
                "query_id": str(row.get("query", total)),
                "query":    query,
                "answer":   ans_text,
                "type":     row.get("type", ""),
            })

        total += 1

    for lang in languages:
        logger.info(f"  {lang}: {len(result[lang])} QA pairs loaded")

    return result


def split_mkqa(
    data: Dict[str, List[Dict]],
    dev_size: int = 500,
    test_size: int = 1000,
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Split into train/dev/test per language.
    dev_size and test_size are taken from the end of the list.
    """
    splits = {}
    for lang, items in data.items():
        n = len(items)
        splits[lang] = {
            "train": items[: n - dev_size - test_size],
            "dev":   items[n - dev_size - test_size : n - test_size],
            "test":  items[n - test_size :],
        }
        logger.info(
            f"  {lang}: train={len(splits[lang]['train'])}  "
            f"dev={len(splits[lang]['dev'])}  test={len(splits[lang]['test'])}"
        )
    return splits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_mkqa()
    splits = split_mkqa(data)
    for lang, s in splits.items():
        sample = s["test"][0]
        print(f"\n[{lang}] sample test query:")
        print(f"  Q: {sample['query']}")
        print(f"  A: {sample['answer']}")
