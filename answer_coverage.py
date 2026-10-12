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
        # Single lowercase mega-string, separator prevents cross-passage false matches.
        mega = "\x00".join(p["text"].lower() for p in corpus)
        covered = 0
        no_gold = 0
        per_item_hit = []
        for item in items:
            answers = [a.strip() for a in gold_answer_strings(item, lang) if a and a.strip()]
            if not answers:
                no_gold += 1
                per_item_hit.append(None)
                continue
            found = any(a.lower() in mega for a in answers)
            per_item_hit.append(found)
            if found:
                covered += 1
        n = len(items)
        denom = n - no_gold
        print(f"{lang}: n={n}, no_gold_answer={no_gold}, covered={covered}/{denom} answerable items, "
              f"coverage_of_answerable={covered/max(1,denom):.4f}, coverage_of_all={covered/n:.4f}  "
              f"[{time.time()-t0:.1f}s elapsed]")
        json.dump({"lang": lang, "n": n, "no_gold_answer": no_gold, "covered": covered,
                    "per_item_hit": per_item_hit},
                   open(f"results/answer_coverage_{lang}.json", "w"))

if __name__ == "__main__":
    main()
