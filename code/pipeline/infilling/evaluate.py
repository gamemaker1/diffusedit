"""Score an infilling run against the gold rewrites of its EDIT sentences.

For every repaired sentence, with the gold rewrite as reference:

    chosen      ROUGE-1/2/L F1 of the output sentence
    copy        the same for the unrepaired source sentence, as a floor
    pool        the best ROUGE-L over every candidate the run generated,
                the oracle reranking of Section 9.3. The gap between pool
                and chosen is what the reranker loses.
    new words   words of the gold rewrite that are not in the source, the
                new facts. `hit` is the fraction of sentences where some
                candidate contains all of them, the Risk 1 measure of
                Section 11. Compare it between runs with and without
                --evidence. `coverage` is the mean fraction covered.

Sentences whose rewrite adds no new words enter the ROUGE numbers only.

Usage:
    python -m infilling.evaluate runs/oracle ../datasets/collated/test.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer

from classifier.inputs import read_jsonl

ROUGE = ("rouge1", "rouge2", "rougeL")


def words(text):
    return set(re.findall(r"\w+", text.lower()))


def gold(record, j, key):
    entry = record["labels"][j]
    if key not in entry:
        raise SystemExit(f"{record['id']}: label entry {j} has no {key!r}; it has {sorted(entry)}, "
                         f"the record has {sorted(record)}. Pass --target-key")
    return entry[key]


def coverage(new, text):
    return len(new & words(text)) / len(new)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("data", help="the collated split the run read, with gold rewrites")
    parser.add_argument("--target-key", default="target", help="gold rewrite field of a label entry")
    parser.add_argument("--out", help="default <run>/metrics.json")
    args = parser.parse_args()

    run = Path(args.run)
    results = list(read_jsonl(run / "results.jsonl"))
    wanted = {r["id"] for r in results}
    records = {r["id"]: r for r in read_jsonl(args.data) if r["id"] in wanted}
    scorer = rouge_scorer.RougeScorer(list(ROUGE), use_stemmer=True)

    rows, stats = [], Counter()
    for result in results:
        record = records[result["id"]]
        stats.update(result["stats"])
        for edit in result["edits"]:
            j = edit["sentence"]
            target = gold(record, j, args.target_key)
            source, output = result["source"][j], result["output"][j]
            texts = [c["text"] for d in edit["decisions"] for c in d["candidates"]]
            chosen = scorer.score(target, output)
            copy = scorer.score(target, source)
            row = {f"chosen_{m}": chosen[m].fmeasure for m in ROUGE}
            row.update({f"copy_{m}": copy[m].fmeasure for m in ROUGE})
            row["pool_rougeL"] = max(scorer.score(target, t)["rougeL"].fmeasure for t in texts)
            new = words(target) - words(source)
            if new:
                row["hit"] = float(any(new <= words(t) for t in texts))
                row["pool_coverage"] = max(coverage(new, t) for t in texts)
                row["chosen_coverage"] = coverage(new, output)
            rows.append(row)

    keys = sorted({k for row in rows for k in row})
    metrics = {k: float(np.mean([row[k] for row in rows if k in row])) for k in keys}
    metrics["sentences"] = len(rows)
    metrics["sentences_with_new_words"] = sum("hit" in row for row in rows)
    metrics["articles"] = len(results)
    if stats["deletion_candidates"]:
        metrics["deletion_win_rate"] = stats["deletion_wins"] / stats["deletion_candidates"]
    metrics["seconds_per_article"] = float(np.mean([r["seconds"] for r in results])) if results else 0.0
    metrics["counters"] = dict(stats)

    for k in keys + ["deletion_win_rate", "seconds_per_article"]:
        if k in metrics:
            print(f"{k:24s} {metrics[k]:.4f}")
    print(f"{metrics['sentences']} sentences ({metrics['sentences_with_new_words']} with new words), "
          f"{metrics['articles']} articles")
    out = Path(args.out) if args.out else run / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
