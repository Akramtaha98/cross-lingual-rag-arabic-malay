"""
run_oracle_gen.py
Uses precomputed oracle passage indices (results/oracle2_precomputed_{lang}.json,
results/oracle2_passages_{lang}.json) to run the corrected oracle-context
experiment (real gold-containing passage vs. closed-book) on a slice of the
covered-item list, for a given language. Appends raw predictions to a JSONL
file so multiple chunked calls accumulate into one dataset; scoring is done
separately once all slices are done.
"""
import argparse, json, time
from generation.mt5_generator import MT5Generator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["ar","ms"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--time_budget", type=float, default=150.0)
    args = ap.parse_args()
    lang = args.lang

    pre = json.load(open(f"results/oracle2_precomputed_{lang}.json", encoding="utf-8"))["items"]
    passmap = json.load(open(f"results/oracle2_passages_{lang}.json", encoding="utf-8"))
    covered = [r for r in pre if r["oracle_idx"] is not None]
    end = args.end if args.end is not None else len(covered)
    slice_ = covered[args.start:end]

    print(f"[{lang}] covered total={len(covered)}, running slice [{args.start}:{end}] = {len(slice_)} items", flush=True)

    t0 = time.time()
    gen = MT5Generator(use_lora=True)
    print(f"model load took {time.time()-t0:.1f}s", flush=True)

    out_path_oracle = f"results/oracle2_gen_oracle_real_{lang}.jsonl"
    out_path_closed = f"results/oracle2_gen_closed_book_{lang}.jsonl"

    f_oracle = open(out_path_oracle, "a", encoding="utf-8")
    f_closed = open(out_path_closed, "a", encoding="utf-8")

    n_done = 0
    tgen0 = time.time()
    for r in slice_:
        if time.time() - tgen0 > args.time_budget:
            print(f"Time budget hit after {n_done} items; stopping early (resume with --start {args.start + n_done}).", flush=True)
            break
        q = r["query"]
        ref = r["ref"]
        passage = passmap[str(r["i"])]

        pred_oracle = gen.generate(q, [(passage, 1.0)], top_k=1)
        pred_closed = gen.generate(q, [], top_k=1)

        f_oracle.write(json.dumps({"i": r["i"], "query": q, "ref": ref, "pred": pred_oracle}, ensure_ascii=False) + "\n")
        f_closed.write(json.dumps({"i": r["i"], "query": q, "ref": ref, "pred": pred_closed}, ensure_ascii=False) + "\n")
        n_done += 1

    f_oracle.close()
    f_closed.close()
    print(f"[{lang}] wrote {n_done} items in {time.time()-tgen0:.1f}s (slice [{args.start}:{args.start+n_done}])", flush=True)

if __name__ == "__main__":
    main()
