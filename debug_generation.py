"""
Diagnose why run_oracle_context.py produced the identical Arabic output
"البحرين" for every query. Run this from cross_lingual_rag/ on your Mac
(same Python env you used for run_all_experiments.py).
"""
from generation.mt5_generator import MT5Generator

gen = MT5Generator(use_lora=True)

tests = [
    ("how long did it take the twin towers to be built",
     [({"text": "The answer is: 11.0 years.", "passage_id": "o"}, 1.0)]),
    ("where did the phrase great scott come from",
     [({"text": "The answer is: unknown.", "passage_id": "o"}, 1.0)]),
    ("how long did it take the twin towers to be built", []),
    (": كم من الوقت استغرق بناء البرجين التوأمين",
     [({"text": "الإجابة هي: 11.0 سنة.", "passage_id": "o"}, 1.0)]),
    (": كم من الوقت استغرق بناء البرجين التوأمين", []),
]

for q, passages in tests:
    beam = gen.generate(q, passages, top_k=max(1, len(passages)))
    greedy = gen.generate(q, passages, top_k=max(1, len(passages)), num_beams=1)
    print("Q:", q)
    print("  beam=4 :", repr(beam))
    print("  greedy :", repr(greedy))
    print()
