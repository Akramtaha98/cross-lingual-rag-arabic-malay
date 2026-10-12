import argparse, json, os, re, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, ".")
from answer_level_metrics import score
ap = argparse.ArgumentParser(); ap.add_argument("--lang", required=True); ap.add_argument("--n", type=int, default=100)
a = ap.parse_args(); L = a.lang
pre = json.load(open(f"results/oracle2_precomputed_{L}.json"))["items"]
pas = json.load(open(f"results/oracle2_passages_{L}.json"))
items = [x for x in pre if x.get("oracle_idx") not in (None, "", "None")][: a.n]
def window(text, ref, w=600):
    t = text.lower(); r = str(ref).lower().strip(); k = t.find(r)
    if k < 0: k = max(0, t.find(re.sub(r"\.0+\b", "", r)))
    s = max(0, k - w); return text[s:s + 2 * w]
def prompt(q, ctx):
    if L == "ar":
        if ctx: return f"أجب عن السؤال باختصار (كلمات قليلة) اعتماداً على المقطع.\n\nالمقطع:\n{ctx}\n\nالسؤال: {q}\nالإجابة:"
        return f"أجب عن السؤال باختصار (كلمات قليلة).\n\nالسؤال: {q}\nالإجابة:"
    if ctx: return f"Jawab soalan dengan ringkas (beberapa perkataan) berdasarkan petikan.\n\nPetikan:\n{ctx}\n\nSoalan: {q}\nJawapan:"
    return f"Jawab soalan dengan ringkas (beberapa perkataan).\n\nSoalan: {q}\nJawapan:"
name = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
jobs = []
for x in items:
    for cond in ("closed_book", "oracle"):
        ctx = window(pas[str(x["i"])]["text"], x["ref"]) if cond == "oracle" else ""
        jobs.append((x, cond, tok.apply_chat_template([{"role": "user", "content": prompt(x["query"], ctx)}], tokenize=False, add_generation_prompt=True)))
out_rows = []; B = 16; t0 = time.time()
for i in range(0, len(jobs), B):
    chunk = jobs[i:i + B]
    enc = tok([c[2] for c in chunk], return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=60, do_sample=False)
    for c, o in zip(chunk, g):
        pred = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        out_rows.append({"i": c[0]["i"], "cond": c[1], "query": c[0]["query"], "ref": c[0]["ref"], "pred": pred})
    print(len(out_rows), "/", len(jobs), round(time.time() - t0), "s", flush=True)
with open(f"results/_qwen_oracle_{L}.jsonl", "w", encoding="utf-8") as f:
    for d in out_rows: f.write(json.dumps(d, ensure_ascii=False) + "\n")
res = {c: score([(d["pred"], d["ref"]) for d in out_rows if d["cond"] == c]) for c in ("closed_book", "oracle")}
json.dump(res, open(f"results/qwen_oracle_{L}.json", "w"), indent=2); print("DONE", L, res)
