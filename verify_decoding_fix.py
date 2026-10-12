"""Quick sanity check: with the new greedy+repetition-penalty decoding config,
do 10 different real MKQA queries per language produce 10 DIFFERENT answers
(as opposed to the old beam=4 collapse to ~2-4 repeated tokens)?
Run from cross_lingual_rag/ on your Mac.
"""
import gzip, json
from generation.mt5_generator import MT5Generator

def load_first_n(path, n=10):
    items = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            items.append(json.loads(line))
    return items

gen = MT5Generator(use_lora=True)
items = load_first_n("indexes/mkqa.jsonl.gz", 10)

for lang in ["ar", "ms"]:
    print(f"--- {lang} ---")
    outputs = []
    for item in items:
        q = item["queries"][lang]
        passages = [({"text": "context placeholder for sanity check.", "passage_id": "x"}, 1.0)]
        ans = gen.generate(q, passages, top_k=1)
        outputs.append(ans)
        print(" ", repr(ans))
    n_unique = len(set(outputs))
    print(f"  {n_unique}/10 unique outputs")
