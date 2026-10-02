"""UpdateROUGE of infilling runs (Section 9.2), with paired tests over articles.

UpdateROUGE scores only the sentences an update touched. For one article, let
U(output) be the output sentences that do not occur verbatim, after whitespace
normalization, in the source article, and U(gold) the same for the gold
article. ROUGE-1, ROUGE-2 and ROUGE-L F1 compare the concatenation of U(output)
with the concatenation of U(gold). An article with an empty U(gold) is
skipped, and an empty U(output) scores 0.

Two gold sides are reported.

    full    every gold sentence absent from the source, including sentences
            the revision added. The pipeline never adds sentences, so this
            caps what it can reach.
    edits   only gold sentences aligned to a source EDIT sentence, the part of
            the update the pipeline attempts

"copy", the unchanged article, has an empty U(output) and scores 0 by
construction. The paired tests of infilling.significance compare runs with the
article as the unit.

Usage:
    python -m infilling.update_rouge ../datasets/collated/train.jsonl \\
        oracle=runs/fix-oracle noev=runs/fix-oracle-noev predicted=runs/fix-predicted \\
        --pairs oracle:noev oracle:predicted
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer

from classifier.inputs import read_jsonl
from infilling.significance import holm, test

ROUGE = ("rouge1", "rouge2", "rougeL")


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("runs", nargs="+", help="name=run_dir")
    parser.add_argument("--pairs", nargs="*", default=[], help="A:B")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="JSON output path")
    args = parser.parse_args()

    runs = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        runs[name] = {r["id"]: r for r in read_jsonl(Path(path) / "results.jsonl")}
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    records = {r["id"]: r for r in read_jsonl(args.data) if r["id"] in set(ids)}
    scorer = rouge_scorer.RougeScorer(list(ROUGE), use_stemmer=True)

    scores = {side: {name: {m: [] for m in ROUGE} for name in runs} for side in ("full", "edits")}
    kept = {side: [] for side in ("full", "edits")}
    for article_id in ids:
        record = records[article_id]
        source = {norm(s) for s in record["source_sentences"]}
        targets = [t["text"] for t in record["target_sentences"]]
        edit_targets = {lab["target_idx"] for lab in record["labels"] if lab["label"] == "EDIT"}
        gold = {"full": [t for t in targets if norm(t) not in source],
                "edits": [t for j, t in enumerate(targets) if j in edit_targets and norm(t) not in source]}
        for side, gold_sentences in gold.items():
            if not gold_sentences:
                continue
            kept[side].append(article_id)
            reference = " ".join(gold_sentences)
            for name, run in runs.items():
                updated = [o for o in run[article_id]["output"] if o and norm(o) not in source]
                result = scorer.score(reference, " ".join(updated)) if updated else None
                for m in ROUGE:
                    scores[side][name][m].append(result[m].fmeasure if result else 0.0)

    rng = np.random.default_rng(args.seed)
    report = {"articles": {side: len(v) for side, v in kept.items()}, "resamples": args.resamples}
    for side in ("full", "edits"):
        n = len(kept[side])
        report[side] = {"means": {name: {m: float(np.mean(v[m])) for m in ROUGE} for name, v in scores[side].items()},
                        "per_article": scores[side], "ids": kept[side]}
        print(f"\nUpdateROUGE, gold side '{side}', {n} articles")
        for name in runs:
            print(f"  {name:12s} " + "  ".join(f"{m} {report[side]['means'][name][m]:.4f}" for m in ROUGE))
        rows = []
        for pair in args.pairs:
            a, b = pair.split(":")
            for m in ROUGE:
                na, nb = np.array(scores[side][a][m]), np.array(scores[side][b][m])
                ones = np.ones(n)
                diff, low, high, p = test(na, ones, nb, ones, rng, args.resamples)
                rows.append({"pair": pair, "metric": m, "diff": diff, "ci_low": low, "ci_high": high, "p": p})
        if rows:
            for row, adjusted in zip(rows, holm(np.array([r["p"] for r in rows]))):
                row["p_holm"] = float(adjusted)
            for r in rows:
                mark = "*" if r["p_holm"] < 0.05 else " "
                print(f"  {r['pair']:18s} {r['metric']:7s} {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
                      f"p {r['p']:.4f} p_holm {r['p_holm']:.4f}{mark}")
        report[side]["tests"] = rows
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
