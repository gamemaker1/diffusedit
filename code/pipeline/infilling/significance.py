"""Paired significance tests between infilling runs on the same articles (Section 9.4).

Every system is scored on the same repaired sentences, joined by (article id,
sentence index). "copy" is the unchanged source sentence. A sentence a run did
not repair counts as copied. Each metric is a ratio of sums over sentences:

    rougeL          mean ROUGE-L F1 against the gold rewrite
    precision, recall, new_recall, stale_kept, unsupported
                    the entity metrics of infilling.entities, micro-averaged

The unit of resampling is the article, since sentences of one article are not
independent. For a pair (A, B) the statistic is metric(A) - metric(B).

    95% CI   paired bootstrap: resample articles with replacement
    p        paired permutation: swap A and B inside each article at random,
             two-sided
    p_holm   Holm correction over every test of one subset table

Usage:
    python -m infilling.significance ../datasets/collated/train.jsonl \\
        oracle=runs/fix-oracle noev=runs/fix-oracle-noev predicted=runs/fix-predicted \\
        --pairs oracle:copy oracle:noev predicted:copy oracle:predicted noev:copy
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer

from classifier.inputs import read_jsonl
from infilling.entities import Extractor, counts

METRICS = ("rougeL", "precision", "recall", "new_recall", "stale_kept", "unsupported")
LOWER_IS_BETTER = {"stale_kept", "unsupported"}


def words(text):
    return set(re.findall(r"\w+", text.lower()))


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def test(num_a, den_a, num_b, den_b, rng, resamples):
    """Difference of ratio-of-sums between A and B, with bootstrap CI and permutation p."""
    def diff(na, da, nb, db):
        with np.errstate(invalid="ignore", divide="ignore"):
            return na.sum(-1) / da.sum(-1) - nb.sum(-1) / db.sum(-1)
    observed = float(diff(num_a, den_a, num_b, den_b))
    n = len(num_a)
    idx = rng.integers(0, n, size=(resamples, n))
    boot = diff(num_a[idx], den_a[idx], num_b[idx], den_b[idx])
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    swap = rng.random((resamples, n)) < 0.5
    perm = diff(np.where(swap, num_b, num_a), np.where(swap, den_b, den_a),
                np.where(swap, num_a, num_b), np.where(swap, den_a, den_b))
    p = (np.sum(np.abs(perm) >= abs(observed) - 1e-12) + 1) / (resamples + 1)
    return observed, float(low), float(high), float(p)


def collect(data, run_dirs):
    """Per-article sums of every metric for "copy" and each run.

    Returns the article ids, the per-subset sentence counts, and
    sums[subset][system][metric], an array [2, n_articles] of summed
    numerators (row 0) and denominators (row 1).
    """
    runs = {name: {r["id"]: r for r in read_jsonl(Path(path) / "results.jsonl")} for name, path in run_dirs.items()}
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    records = {r["id"]: r for r in read_jsonl(data) if r["id"] in set(ids)}
    m, scorer = Extractor(), rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    systems = ["copy", *runs]
    sums = {sub: {s: {k: np.zeros((2, len(ids))) for k in METRICS} for s in systems} for sub in ("all", "supported")}
    n_sentences = defaultdict(int)
    for a, article_id in enumerate(ids):
        record = records[article_id]
        e = set().union(*(m(f"{s['title']}: {s['text']}") for s in record["evidence"])) if record["evidence"] else set()
        evidence_words = words(" ".join(f"{s['title']} {s['text']}" for s in record["evidence"]))
        repaired = sorted({edit["sentence"] for r in runs.values() for edit in r[article_id]["edits"]})
        for j in repaired:
            entry = record["labels"][j]
            target = record["target_sentences"][entry["target_idx"]]["text"]
            source = record["source_sentences"][j].strip()
            g, s = m(target), m(source)
            new_words = words(target) - words(source)
            subsets = ["all"] + (["supported"] if new_words and new_words <= evidence_words | words(source) else [])
            for sub in subsets:
                n_sentences[sub] += 1
            outputs = {"copy": source, **{name: r[article_id]["output"][j] for name, r in runs.items()}}
            for system, text in outputs.items():
                c = counts(m(text), g, s, e)
                values = {k: (c[f"{k}_num"], c[f"{k}_den"]) for k in METRICS if k != "rougeL"}
                values["rougeL"] = (scorer.score(target, text)["rougeL"].fmeasure, 1)
                for sub in subsets:
                    for k, (num, den) in values.items():
                        sums[sub][system][k][:, a] += (num, den)
    return ids, dict(n_sentences), sums


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("runs", nargs="+", help="name=run_dir")
    parser.add_argument("--pairs", nargs="+", required=True, help="A:B, where copy is the unchanged sentence")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="JSON output path")
    args = parser.parse_args()

    ids, n_sentences, sums = collect(args.data, dict(spec.split("=", 1) for spec in args.runs))

    rng = np.random.default_rng(args.seed)
    report = {"articles": len(ids), "sentences": dict(n_sentences), "resamples": args.resamples, "tables": {}}
    for sub in ("all", "supported"):
        rows = []
        for pair in args.pairs:
            a_name, b_name = pair.split(":")
            for k in METRICS:
                (na, da), (nb, db) = sums[sub][a_name][k], sums[sub][b_name][k]
                keep = (da > 0) | (db > 0)
                diff, low, high, p = test(na[keep], da[keep], nb[keep], db[keep], rng, args.resamples)
                rows.append({"pair": pair, "metric": k, "a": float(na.sum() / da.sum()) if da.sum() else None,
                             "b": float(nb.sum() / db.sum()) if db.sum() else None,
                             "diff": diff, "ci_low": low, "ci_high": high, "p": p})
        for row, adjusted in zip(rows, holm(np.array([r["p"] for r in rows]))):
            row["p_holm"] = float(adjusted)
        report["tables"][sub] = rows
        print(f"\n{sub}: {n_sentences[sub]} sentences in {len(ids)} articles")
        print(f"{'pair':22s} {'metric':12s} {'A':>6s} {'B':>6s} {'A-B':>7s} {'95% CI':>17s} {'p':>7s} {'p_holm':>7s}")
        for r in rows:
            mark = "*" if r["p_holm"] < 0.05 else " "
            print(f"{r['pair']:22s} {r['metric']:12s} {r['a']:6.3f} {r['b']:6.3f} {r['diff']:+7.3f} "
                  f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] {r['p']:7.4f} {r['p_holm']:7.4f}{mark}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
