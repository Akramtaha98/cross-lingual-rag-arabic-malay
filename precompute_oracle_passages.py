import gzip, json, time

def load_mkqa_first_n(path, n=1000):
    items = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            items.append(json.loads(line))
    return items

def gold_answer_strings(item, lang):
    ans_list = item.get("answers", {}).get(lang, [])
    out = []
    for a in ans_list:
        t = a.get("text")
        if t:
            out.append(t)
        for alias in a.get("aliases", []) or []:
            if alias:
                out.append(alias)
    return out

def main():
    t0 = time.time()
    items = load_mkqa_first_n("indexes/mkqa.jsonl.gz", 1000)
    for lang, corpus_file in [("ar", "indexes/corpus_ar.json"), ("ms", "indexes/corpus_ms.json")]:
        corpus = json.load(open(corpus_file, encoding="utf-8"))
        lowered = [p["text"].lower() for p in corpus]
        results = []
        for i, item in enumerate(items):
            q = item["queries"].get(lang)
            answers = [a.strip() for a in gold_answer_strings(item, lang) if a and a.strip()]
            ref = answers[0] if answers else ""
            oracle_idx = None
            for a in answers:
                al = a.lower()
                for j, lt in enumerate(lowered):
                    if al in lt:
                        oracle_idx = j
                        break
                if oracle_idx is not None:
                    break
            results.append({"i": i, "query": q, "ref": ref, "oracle_idx": oracle_idx})
        n_covered = sum(1 for r in results if r["oracle_idx"] is not None)
        print(f"{lang}: {n_covered}/{len(items)} covered  [{time.time()-t0:.1f}s]")
        json.dump({"lang": lang, "items": results}, open(f"results/oracle2_precomputed_{lang}.json", "w", encoding="utf-8"), ensure_ascii=False)
        # also dump the found passages themselves for use during generation
        passmap = {str(r["i"]): corpus[r["oracle_idx"]] for r in results if r["oracle_idx"] is not None}
        json.dump(passmap, open(f"results/oracle2_passages_{lang}.json", "w", encoding="utf-8"), ensure_ascii=False)

if __name__ == "__main__":
    main()
