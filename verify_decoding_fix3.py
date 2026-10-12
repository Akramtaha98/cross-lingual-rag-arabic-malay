"""Final check: with the patched generator (config-driven sampling + seeded
per-call), do (a) 10 different AR/MS queries give varied outputs, and
(b) does rerunning the SAME query give the SAME output (reproducibility)?
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

items = load_first_n("indexes/mkqa.jsonl.gz", 10)
passage = [({"text": "context placeholder for sanity check.", "passage_id": "x"}, 1.0)]

gen = MT5Generator(use_lora=True)
for lang in ["ar", "ms"]:
    print(f"--- {lang} (run 1) ---")
    outs1 = []
    for item in items:
        a = gen.generate(item["queries"][lang], passage, top_k=1)
        outs1.append(a)
        print(" ", repr(a))
    print(f"  {len(set(outs1))}/10 unique")

# Reproducibility check: fresh generator instance, same first query, same seed state
gen2 = MT5Generator(use_lora=True)
a_repeat = gen2.generate(items[0]["queries"]["ar"], passage, top_k=1)
print("\nReproducibility check (first AR query, fresh instance):", repr(a_repeat))
