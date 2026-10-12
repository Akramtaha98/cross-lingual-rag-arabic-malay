"""
Isolate whether the collapse comes from (a) the base mT5-small model itself,
or (b) the LoRA adapter / merge step, or (c) a peaked-but-not-degenerate
distribution that sampling can escape.
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

print("=== BASE mT5-small, no LoRA, greedy ===")
gen_base = MT5Generator(use_lora=False)
outs = []
for item in items:
    a = gen_base.generate(item["queries"]["ar"], passage, top_k=1, num_beams=1)
    outs.append(a)
    print(" ", repr(a))
print(f"  {len(set(outs))}/10 unique\n")

print("=== LoRA merged, SAMPLING (temperature=0.9, top_p=0.9) ===")
gen_lora = MT5Generator(use_lora=True)
outs = []
for item in items:
    a = gen_lora.generate(item["queries"]["ar"], passage, top_k=1,
                           do_sample=True, temperature=0.9, top_p=0.9, num_beams=1)
    outs.append(a)
    print(" ", repr(a))
print(f"  {len(set(outs))}/10 unique\n")

print("=== LoRA merged, greedy, LONGER max_new_tokens=32, no repetition penalty ===")
outs = []
for item in items:
    a = gen_lora.generate(item["queries"]["ar"], passage, top_k=1,
                           num_beams=1, max_new_tokens=32, repetition_penalty=1.0, no_repeat_ngram_size=0)
    outs.append(a)
    print(" ", repr(a))
print(f"  {len(set(outs))}/10 unique\n")
