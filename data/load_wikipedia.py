"""
data/load_wikipedia.py
Build chunked passage corpora from Arabic and Malay Wikipedia.

Uses HuggingFace's "wikipedia" dataset (20231101 snapshot).
Chunks each article into overlapping windows of ~200 words.
"""

import re
import os
import json
import logging
from typing import List, Dict, Optional
from datasets import load_dataset
from tqdm import tqdm
from config import WIKI_CHUNK_SIZE, WIKI_CHUNK_OVERLAP, WIKI_MAX_PASSAGES, INDEX_DIR

logger = logging.getLogger(__name__)

WIKI_DATE = "20231101"  # Wikipedia snapshot date


def clean_text(text: str) -> str:
    """Basic Wikipedia article cleanup."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\[\d+\]", "", text)        # remove citation markers [1]
    text = re.sub(r"={2,}.*?={2,}", "", text)  # remove section headers ==...==
    text = text.strip()
    return text


def chunk_text(
    text: str,
    chunk_size: int = WIKI_CHUNK_SIZE,
    overlap: int = WIKI_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping word-level chunks.
    chunk_size: target words per chunk
    overlap: words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.split()) >= 20:           # skip very short trailing chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_corpus(
    language: str,
    max_passages: int = WIKI_MAX_PASSAGES,
    cache_path: Optional[str] = None,
) -> List[Dict]:
    """
    Build a list of passage dicts:
      {"passage_id": "ar_0001_chunk_3",
       "text": "...",
       "title": "article title",
       "lang": "ar"}

    Caches to JSON at cache_path to avoid re-downloading.
    """
    if cache_path is None:
        os.makedirs(INDEX_DIR, exist_ok=True)
        cache_path = os.path.join(INDEX_DIR, f"corpus_{language}.json")

    if os.path.exists(cache_path):
        logger.info(f"Loading cached corpus from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        logger.info(f"  {language}: {len(corpus)} passages loaded from cache")
        return corpus

    logger.info(f"Building corpus for language={language} …")
    wiki = load_dataset(
        "wikimedia/wikipedia",
        f"{WIKI_DATE}.{language}",
    )["train"]

    corpus = []
    passage_count = 0

    for i, article in enumerate(tqdm(wiki, desc=f"Chunking {language} Wikipedia")):
        if passage_count >= max_passages:
            break

        text = clean_text(article["text"])
        title = article["title"]
        chunks = chunk_text(text)

        for j, chunk in enumerate(chunks):
            if passage_count >= max_passages:
                break
            corpus.append({
                "passage_id": f"{language}_{i:06d}_chunk_{j:03d}",
                "text":       chunk,
                "title":      title,
                "lang":       language,
            })
            passage_count += 1

    logger.info(f"  {language}: {len(corpus)} passages built from {i+1} articles")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    logger.info(f"  Saved corpus to {cache_path}")

    return corpus


def load_both_corpora(
    max_passages_per_lang: int = WIKI_MAX_PASSAGES,
) -> Dict[str, List[Dict]]:
    """Load Arabic and Malay corpora, return as dict keyed by language code."""
    return {
        "ar": build_corpus("ar", max_passages_per_lang),
        "ms": build_corpus("ms", max_passages_per_lang),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    corpora = load_both_corpora(max_passages_per_lang=10_000)
    for lang, passages in corpora.items():
        print(f"\n[{lang}] {len(passages)} passages. Sample:")
        print(f"  Title: {passages[0]['title']}")
        print(f"  Text:  {passages[0]['text'][:200]}…")
