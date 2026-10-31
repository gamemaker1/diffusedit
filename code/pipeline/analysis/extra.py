"""Sentence scores of any runs on any data file, on the same gold EDIT sentences.

For each run, every gold EDIT sentence of the articles it read is scored
against the gold rewrite: ROUGE-L of the output and of the unchanged source,
exact match, new-value recall and the share of old values kept, with the
article as the unit of the bootstrap interval. An editor run, which logs
whole articles and no candidates, is scored the same way as an infilling run.
With --ids only those articles are read, and with --supported only sentences
whose new words all occur in the evidence or the source are scored.

Usage:
    python -m analysis.extra DATA name=RUN_DIR ... [--ids IDS] [--supported]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer

from analysis.values import holds, norm
from classifier.inputs import read_jsonl
from experiments.stress import gold_spans
from infilling.evaluate import words

SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def supported(record, j):
    entry = record["labels"][j]
    new = words(entry["target"]) - words(record["source_sentences"][j])
    have = words(" ".join(f"{p['title']} {p['text']}" for p in record["evidence"])) | words(record["source_sentences"][j])
    return new <= have


def score(records, results, only_supported):
    rows = {}
    for result in results:
        record = records[result["id"]]
        for j, entry in enumerate(record["labels"]):
            if entry.get("label") != "EDIT" or (only_supported and not supported(record, j)):
                continue
            source, out, target = record["source_sentences"][j], result["output"][j], entry["target"]
            pairs = gold_spans(source, target, entry["stale_spans"])
            new = [n for _, n in pairs if n.strip()]
            old = [o for o, _ in pairs if o.strip() and not holds(target, o)]
            rows.setdefault(result["id"], []).append({
                "chosen": SCORER.score(target, out)["rougeL"].fmeasure,
                "copy": SCORER.score(target, source)["rougeL"].fmeasure,
                "exact": float(norm(out) == norm(target)),
                "new": np.mean([holds(out, n) for n in new]) if new else np.nan,
                "old": np.mean([holds(out, o) for o in old]) if old else np.nan,
                "changed": float(out != source)})
    return rows


def ratio(rows, key, resamples=2000, seed=0):
    s = np.array([np.nansum([r[key] for r in rs]) for rs in rows.values()])
    c = np.array([np.sum([not np.isnan(r[key]) for r in rs]) for rs in rows.values()])
    idx = np.random.default_rng(seed).integers(0, len(s), (resamples, len(s)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("runs", nargs="+", help="name=run_dir")
    parser.add_argument("--ids", help="JSON list of article ids")
    parser.add_argument("--supported", action="store_true")
    args = parser.parse_args()
    keep = set(json.loads(Path(args.ids).read_text())) if args.ids else None
    records = {r["id"]: r for r in read_jsonl(args.data) if keep is None or r["id"] in keep}
    print(f"{'run':14s} {'sent':>5s} {'copy':>6s} {'rougeL':>6s} {'[int]':>16s} {'exact':>6s} {'new':>6s} {'old':>6s} {'changed':>7s}")
    for item in args.runs:
        name, path = item.split("=", 1)
        results = [r for r in read_jsonl(Path(path) / "results.jsonl") if r["id"] in records]
        rows = score(records, results, args.supported)
        n = sum(len(v) for v in rows.values())
        m, lo, hi = ratio(rows, "chosen")
        print(f"{name:14s} {n:5d} {ratio(rows, 'copy')[0]:6.3f} {m:6.3f} [{lo:6.3f}, {hi:6.3f}] "
              f"{ratio(rows, 'exact')[0]:6.3f} {ratio(rows, 'new')[0]:6.3f} {ratio(rows, 'old')[0]:6.3f} {ratio(rows, 'changed')[0]:7.3f}")


if __name__ == "__main__":
    main()
