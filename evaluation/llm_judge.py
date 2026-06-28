"""
evaluation/llm_judge.py
=======================
LLM-as-judge faithfulness evaluator using the Anthropic API.

Given a (query, retrieved_passages, generated_answer) triple, the judge LLM
scores how faithful the answer is to the retrieved context on a 0–5 scale,
which is then normalised to [0, 1].

This replaces the simple n-gram overlap "faithfulness" metric from metrics.py
with a semantically aware evaluator — accepted practice in recent NLP papers
(e.g. RAGAS, TruLens, ARES).

Usage
-----
    from evaluation.llm_judge import LLMJudge

    judge = LLMJudge()                      # uses claude-haiku-4-5 by default (cheap)
    score = judge.score(query, passages, answer)          # single item → float
    scores = judge.batch_score(items, batch_size=10)     # list of dicts → list of floats

Environment
-----------
    export ANTHROPIC_API_KEY="sk-ant-..."

Cost estimate (claude-haiku-4-5, 2025 pricing):
    ~200 input tokens + ~20 output tokens per item
    1,000 items × 2 languages = 2,000 calls ≈ $0.05–0.10 total
"""

import os
import json
import logging
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """\
You are an expert evaluator for question-answering systems.

Your task: assess whether the GENERATED ANSWER is faithful to the RETRIEVED CONTEXT.
A faithful answer only contains information that is directly supported by the retrieved context.
An unfaithful answer contains hallucinations, contradictions, or information not in the context.

QUESTION: {query}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER: {answer}

Score the faithfulness on a scale of 0 to 5:
0 = Completely unfaithful / hallucinated / empty
1 = Mostly unfaithful, minor overlap with context
2 = Partially faithful, some unsupported claims
3 = Mostly faithful, minor unsupported details
4 = Largely faithful, very minor omissions or paraphrasing
5 = Completely faithful, fully supported by the context

Respond with ONLY a JSON object in this format (no other text):
{{"score": <integer 0-5>, "reason": "<one sentence>"}}
"""

# ── Judge class ───────────────────────────────────────────────────────────────

class LLMJudge:
    """
    Faithfulness evaluator backed by an Anthropic model (default: claude-haiku-4-5).

    Parameters
    ----------
    model : str
        Anthropic model ID. claude-haiku-4-5 is cheap and sufficient for scoring.
        Use claude-sonnet-4-6 for higher quality at higher cost.
    max_context_tokens : int
        Truncate retrieved context to this many characters to stay within token limits.
    retry_delay : float
        Seconds to wait between retries on API errors.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_context_chars: int = 2000,
        retry_delay: float = 2.0,
    ):
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic --break-system-packages"
            )
        self.model = model
        self.max_context_chars = max_context_chars
        self.retry_delay = retry_delay
        logger.info(f"LLMJudge initialised (model={model})")

    def _build_context(self, passages) -> str:
        """
        Format retrieved passages into a single context string.
        passages: list of (passage_dict, score) tuples OR list of passage_dicts
        """
        parts = []
        for i, item in enumerate(passages, 1):
            if isinstance(item, tuple):
                passage, score = item
            else:
                passage = item
            text = passage.get("text", "")
            parts.append(f"[{i}] {text}")
        combined = "\n\n".join(parts)
        # Truncate to avoid token limits
        return combined[: self.max_context_chars]

    def score(
        self,
        query: str,
        passages,
        answer: str,
        retries: int = 3,
    ) -> float:
        """
        Score a single (query, passages, answer) triple.
        Returns a float in [0, 1] (raw score / 5).
        Returns 0.0 on failure after retries.
        """
        if not answer or answer.strip() == "":
            return 0.0

        context = self._build_context(passages)
        prompt = FAITHFULNESS_PROMPT.format(
            query=query,
            context=context,
            answer=answer,
        )

        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=64,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                data = json.loads(text)
                raw_score = int(data["score"])
                raw_score = max(0, min(5, raw_score))   # clamp to [0, 5]
                return raw_score / 5.0
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Parse error on attempt {attempt+1}: {e}")
                if attempt < retries - 1:
                    time.sleep(self.retry_delay)
            except Exception as e:
                logger.warning(f"API error on attempt {attempt+1}: {e}")
                if attempt < retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        logger.error(f"LLM judge failed after {retries} retries. Returning 0.0.")
        return 0.0

    def batch_score(
        self,
        items: List[Dict],
        batch_size: int = 10,
    ) -> List[float]:
        """
        Score a list of items.

        Each item must be a dict with keys:
            "query"    : str
            "passages" : list of (passage_dict, score) tuples
            "answer"   : str

        Returns a list of float scores in [0, 1].
        """
        scores = []
        for i, item in enumerate(items):
            score = self.score(
                query=item["query"],
                passages=item["passages"],
                answer=item["answer"],
            )
            scores.append(score)
            if (i + 1) % batch_size == 0:
                logger.info(f"LLM judge: {i+1}/{len(items)} scored, "
                            f"running mean={sum(scores)/len(scores):.3f}")
            # Polite rate limiting
            time.sleep(0.1)
        return scores

    def evaluate_experiment(
        self,
        results: List[Dict],
        lang: str = "ar",
    ) -> Dict:
        """
        Run LLM-judge faithfulness evaluation on a full experiment result list.
        results: output from pipeline.run_batch()

        Returns a dict with:
            llm_faithfulness_mean  : float
            llm_faithfulness_scores: list of float
            llm_faithfulness_lang  : str
        """
        items = [
            {
                "query":    r["query"],
                "passages": r["passages"],
                "answer":   r["answer"],
            }
            for r in results
        ]
        scores = self.batch_score(items)
        mean_score = sum(scores) / len(scores) if scores else 0.0

        logger.info(f"LLM faithfulness ({lang}): mean={mean_score:.4f} "
                    f"over {len(scores)} items")

        return {
            "llm_faithfulness_mean":   mean_score,
            "llm_faithfulness_scores": scores,
            "llm_faithfulness_lang":   lang,
        }


# ── Standalone runner ─────────────────────────────────────────────────────────

def run_llm_judge_on_results(
    results_dir: str = "./results",
    output_path: str = "./results/llm_judge_scores.json",
    model: str = "claude-haiku-4-5-20251001",
) -> None:
    """
    Load all experiment result JSONs and compute LLM-judge faithfulness scores.
    Saves a JSON with per-experiment mean scores.

    Usage:
        python -c "from evaluation.llm_judge import run_llm_judge_on_results; run_llm_judge_on_results()"
    """
    import glob, os

    judge = LLMJudge(model=model)
    all_scores = {}

    result_files = sorted(glob.glob(os.path.join(results_dir, "*_results.json")))
    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    for path in result_files:
        exp_name = os.path.basename(path).replace("_results.json", "")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        samples = data.get("samples", [])
        if not samples:
            logger.warning(f"No samples in {exp_name}, skipping LLM judge.")
            continue

        lang = data.get("config", {}).get("query_lang", "ar")
        eval_result = judge.evaluate_experiment(samples, lang=lang)
        all_scores[exp_name] = eval_result
        print(f"  {exp_name}: LLM faithfulness = {eval_result['llm_faithfulness_mean']:.4f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)
    print(f"\nSaved LLM judge scores → {output_path}")


if __name__ == "__main__":
    run_llm_judge_on_results()
