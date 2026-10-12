import torch
from generation.mt5_generator import MT5Generator, build_input

gen = MT5Generator(use_lora=True)

q = "كم من الوقت استغرق بناء البرجين التوأمين"
contexts = {
    "empty": [],
    "oracle": [({"text": "الإجابة هي: 11.0 سنة.", "passage_id":"o"}, 1.0)],
    "real_dummy": [({"text": "برج بتروناس التوأم في كوالالمبور استغرق بناؤه 11 سنة وانتهى عام 1998.", "passage_id":"r"}, 1.0)],
}
for name, ctx in contexts.items():
    print("=== context:", name, "===")
    input_text = build_input(q, ctx, max_passages=1)
    print("input:", input_text)
    inputs = gen.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(gen.device)
    for seed in [42, 43, 44]:
        torch.manual_seed(seed)
        with torch.no_grad():
            out = gen.model.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=0.9, top_p=0.9, num_beams=1, repetition_penalty=1.3, no_repeat_ngram_size=3, decoder_start_token_id=gen.tokenizer.pad_token_id)
        txt = gen.tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  seed={seed}: {txt!r}")
