"""Value-level scores of the EDIT sentences of a run, with article bootstrap intervals.

ROUGE-L of a whole sentence is high for any output that keeps most of the
source, so copying already scores 0.875 on the synthetic test set. These
scores look only at the swapped values. For each gold EDIT sentence, with the
(old, new) value pairs of its stale spans (experiments.stress.gold_spans):

    exact       the output equals the gold update, ignoring case and spacing
    new         the share of new values that occur in the output as whole words
    old kept    the share of old values that still occur in the output, over
                old values that do not also occur in the gold update

Every gold EDIT sentence of the articles a run read is scored, and a sentence
the run did not change counts as its source. "copy" scores the source
sentences of the articles in --ids.

Usage:
    python -m analysis.values ../datasets/synthetic/split/test.jsonl \\
        outputs/jl/full/runs/fix-tuned outputs/jl/full/runs/editor copy \\
        --ids outputs/jl/full/supported_ids.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.stress import gold_spans, holds

KEYS = ("exact", "new", "old kept")


def norm(text):
    return " ".join(text.split()).lower()


def score(record, outputs):
    """One (exact, new, old kept) row per gold EDIT sentence, NaN where a share is undefined."""
    rows = []
    for j, entry in enumerate(record["labels"]):
        if entry.get("label") != "EDIT":
            continue
        out = outputs[j]
        pairs = gold_spans(record["source_sentences"][j], entry["target"], entry["stale_spans"])
        new = [n for _, n in pairs if n.strip()]
        old = [o for o, _ in pairs if o.strip() and not holds(entry["target"], o)]
        rows.append((float(norm(out) == norm(entry["target"])),
                     np.mean([holds(out, n) for n in new]) if new else np.nan,
                     np.mean([holds(out, o) for o in old]) if old else np.nan))
    return rows


def interval(per_article, k, resamples=2000, seed=0):
    """Ratio-of-sums mean of column k and its 95% bootstrap interval over articles."""
    s = np.array([np.nansum([r[k] for r in rows]) for rows in per_article])
    c = np.array([np.sum([not np.isnan(r[k]) for r in rows]) for rows in per_article])
    idx = np.random.default_rng(seed).integers(0, len(s), (resamples, len(s)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), *np.percentile(means, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="the dataset split the runs read")
    parser.add_argument("runs", nargs="+", help="run directories with results.jsonl, or copy")
    parser.add_argument("--ids", required=True, help="JSON list of the article ids to score")
    args = parser.parse_args()

    ids = set(json.loads(Path(args.ids).read_text()))
    records = {}
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in ids:
                records[r["id"]] = r
    print(f"{'run':24s}" + "".join(f"{k:>26s}" for k in KEYS))
    for run in args.runs:
        if run == "copy":
            per = [score(r, r["source_sentences"]) for r in records.values()]
        else:
            per = []
            with open(Path(run) / "results.jsonl") as f:
                for line in f:
                    res = json.loads(line)
                    if res["id"] in records:
                        per.append(score(records[res["id"]], res["output"]))
        cells = [interval(per, k) for k in range(len(KEYS))]
        print(f"{Path(run).name:24s}" + "".join(f"{m:10.3f} [{lo:.3f}, {hi:.3f}]" for m, lo, hi in cells))


if __name__ == "__main__":
    main()
