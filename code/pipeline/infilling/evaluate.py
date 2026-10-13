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

Every metric is also reported on two subsets of the sentences.

    supported   every new word of the gold rewrite appears in the evidence or
                the source sentence, so the input contains the new facts
    cited       the gold target sentence cites an evidence snippet
                (FRUIT's evidence_refs), when the data carries the field

masked_fraction is the share of source characters the run masked, and
stale_fraction the share the derived labels mark stale.

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
    if key not in entry and entry.get("target_idx") is not None:
        return record["target_sentences"][entry["target_idx"]]["text"]
    if key not in entry:
        raise SystemExit(f"{record['id']}: label entry {j} has no {key!r}; it has {sorted(entry)}, "
                         f"the record has {sorted(record)}. Pass --target-key")
    return entry[key]


def coverage(new, text):
    return len(new & words(text)) / len(new)


def summarize(rows):
    keys = sorted({k for row in rows for k in row if not k.startswith("is_")})
    metrics = {k: float(np.mean([row[k] for row in rows if k in row])) for k in keys}
    metrics["sentences"] = len(rows)
    metrics["sentences_with_new_words"] = sum("hit" in row for row in rows)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("data", help="the collated split the run read, with gold rewrites")
    parser.add_argument("--target-key", default="target", help="gold rewrite field of a label entry")
    parser.add_argument("--limit", type=int, help="score only the first N articles of the run")
    parser.add_argument("--out", help="default <run>/metrics.json")
    args = parser.parse_args()

    run = Path(args.run)
    results = list(read_jsonl(run / "results.jsonl"))[:args.limit]
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
            entry = record["labels"][j]
            evidence = words(" ".join(f"{s['title']} {s['text']}" for s in record["evidence"]))
            texts = [c["text"] for d in edit["decisions"] for c in d["candidates"]]
            chosen = scorer.score(target, output)
            copy = scorer.score(target, source)
            row = {f"chosen_{m}": chosen[m].fmeasure for m in ROUGE}
            row.update({f"copy_{m}": copy[m].fmeasure for m in ROUGE})
            row["pool_rougeL"] = max(scorer.score(target, t)["rougeL"].fmeasure for t in texts)
            new = words(target) - words(source)
            row["masked_fraction"] = sum(len(m) for m in edit.get("masked", [])) / max(len(source), 1)
            if "stale_spans" in entry:
                row["stale_fraction"] = sum(b - a for a, b in entry["stale_spans"]) / max(len(source), 1)
            row["is_supported"] = bool(new) and new <= evidence | words(source)
            if entry.get("target_idx") is not None and "target_sentences" in record:
                row["is_cited"] = bool(record["target_sentences"][entry["target_idx"]].get("evidence_refs"))
            if new:
                row["hit"] = float(any(new <= words(t) for t in texts))
                row["pool_coverage"] = max(coverage(new, t) for t in texts)
                row["chosen_coverage"] = coverage(new, output)
            rows.append(row)

    metrics = summarize(rows)
    keys = [k for k in metrics if k not in ("sentences", "sentences_with_new_words")]
    for subset in ("supported", "cited"):
        part = [row for row in rows if row.get(f"is_{subset}")]
        if part:
            metrics[subset] = summarize(part)
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
    for subset in ("supported", "cited"):
        if subset in metrics:
            m = metrics[subset]
            print(f"{subset}: {m['sentences']} sentences  " + "  ".join(
                f"{k} {m[k]:.4f}" for k in ("chosen_rougeL", "copy_rougeL", "pool_rougeL", "hit",
                                             "pool_coverage", "chosen_coverage") if k in m))
    out = Path(args.out) if args.out else run / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
