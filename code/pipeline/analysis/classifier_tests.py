"""Paired tests between sentence classifiers on the same held-out sentences.

Every configuration of one seed holds out the same articles, so their
heldout_predictions.npz files cover the same sentences. For a pair (A, B) the
statistic is macro-F1(A) - macro-F1(B), each with its own tuned DROP gate.

    seed 42   paired bootstrap over held-out articles (95% interval) and a
              paired permutation test that swaps the two predictions inside
              each article, 2,000 resamples each
    5 seeds   the difference on each seed's own held-out split, its mean and
              standard deviation

Holm correction runs over the pairs.

--file test_predictions.npz compares the heads on the test split instead of
the held-out split. --baseline points at the MiniLM seeds when they are not in
the sweep, and --noev adds the block-24 head on states without evidence.
--pooling adds the block-24 evidence-augmented heads of
experiments.evidence_train (mean, nearest and attention pooling), each against
ce24. They are single runs with seed 42, read from
<pooling>/evidence_<pool>/layer24/<file>, so their rows carry the seed-42 test
and no per-seed spread. A missing pooling file is reported and left out.

Usage:
    python -m analysis.classifier_tests --sweep outputs/jl/heads/sweep --out outputs/jl/heads/sweep/tests.json
    python -m analysis.classifier_tests --sweep full/heads/sweep --baseline full/heads/baseline \\
        --noev full/noev/heads/sweep --pooling full/heads --file test_predictions.npz \
        --out full/analysis/classifier_tests.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from infilling.significance import holm

SEEDS = (42, 1, 2, 3, 4)
DROP, EDIT = 2, 1


def macro_f1(y, pred, weight=None):
    """Macro-F1, with an optional weight per sentence (a bootstrap count)."""
    w = np.ones(len(y)) if weight is None else weight
    f1 = []
    for c in range(3):
        tp = np.sum(w * ((pred == c) & (y == c)))
        fp = np.sum(w * ((pred == c) & (y != c)))
        fn = np.sum(w * ((pred != c) & (y == c)))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1))


def decide(probs, tau):
    pred = probs.argmax(1)
    pred[(pred == DROP) & (probs[:, DROP] < tau)] = EDIT
    return pred


def load(path):
    d = np.load(path)
    return d["rows"], decide(d["probs"], float(d["tau_d"])), d["labels"].astype(np.int64)


def articles_of(path):
    d = np.load(path)
    return d["article"] if "article" in d.files else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--baseline", help="directory of the baseline seeds; default <sweep>/baseline")
    parser.add_argument("--noev", help="sweep directory of the heads on states without evidence")
    parser.add_argument("--pooling", help="heads directory holding evidence_{mean,nearest,attention}/layer24 (single runs)")
    parser.add_argument("--file", default="heldout_predictions.npz", help="prediction file name in each seed directory")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--out")
    args = parser.parse_args()
    sweep = Path(args.sweep)
    configs = {"baseline": Path(args.baseline) if args.baseline else sweep / "baseline", "ce24": sweep / "cls-ce/layer24", "ce32": sweep / "cls-ce/layer32",
               "focal24": sweep / "cls-focal/layer24", "focal32": sweep / "cls-focal/layer32"}
    pairs = [("ce24", "baseline"), ("ce24", "ce32"), ("ce24", "focal24"), ("ce32", "focal32")]
    if args.noev:
        configs["noev24"] = Path(args.noev) / "cls-ce/layer24"
        pairs.append(("ce24", "noev24"))
    # Single runs keep their prediction file in the configuration directory, not in seed<N>.
    single = set()
    if args.pooling:
        for pool in ("mean", "nearest", "attention"):
            path = Path(args.pooling) / f"evidence_{pool}/layer24"
            if (path / args.file).exists():
                configs[f"{pool}24"] = path
                single.add(f"{pool}24")
                pairs.append((f"{pool}24", "ce24"))
            else:
                print(f"{pool}24: {path / args.file} is missing, left out")
    rng = np.random.default_rng(0)

    def seed_file(name, seed):
        if name in single:
            return configs[name] / args.file if seed == 42 else None
        return configs[name] / f"seed{seed}" / args.file

    article = articles_of(seed_file("ce24", 42))
    preds = {}
    for name in configs:
        rows, pred, y = load(seed_file(name, 42))
        preds[name] = (rows, pred, y)
    rows0, _, y = preds["ce24"]
    for name, (rows, _, yy) in preds.items():
        if not (np.array_equal(rows, rows0) and np.array_equal(yy, y)):
            raise SystemExit(f"{name}: held-out rows differ from ce24 for seed 42")
    _, group = np.unique(article, return_inverse=True)
    n_groups = group.max() + 1

    results = []
    for a_name, b_name in pairs:
        pa, pb = preds[a_name][1], preds[b_name][1]
        observed = macro_f1(y, pa) - macro_f1(y, pb)
        boot, perm = [], []
        for _ in range(args.resamples):
            weight = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)[group]
            boot.append(macro_f1(y, pa, weight) - macro_f1(y, pb, weight))
            swap = (rng.random(n_groups) < 0.5)[group]
            qa, qb = np.where(swap, pb, pa), np.where(swap, pa, pb)
            perm.append(macro_f1(y, qa) - macro_f1(y, qb))
        low, high = np.percentile(boot, [2.5, 97.5])
        p = (np.sum(np.abs(perm) >= abs(observed) - 1e-12) + 1) / (args.resamples + 1)
        result = {"pair": f"{a_name}:{b_name}", "diff_seed42": observed, "ci_low": float(low),
                  "ci_high": float(high), "p": float(p)}
        files = [(seed_file(a_name, s), seed_file(b_name, s)) for s in SEEDS]
        if all(fa and fb and fa.exists() and fb.exists() for fa, fb in files):
            per_seed = []
            for fa, fb in files:
                _, pred_a, ya = load(fa)
                _, pred_b, _ = load(fb)
                per_seed.append(macro_f1(ya, pred_a) - macro_f1(ya, pred_b))
            result.update(per_seed=per_seed, seed_mean=float(np.mean(per_seed)),
                          seed_sd=float(np.std(per_seed, ddof=1)))
        results.append(result)
    for r, adjusted in zip(results, holm(np.array([r["p"] for r in results]))):
        r["p_holm"] = float(adjusted)
    for r in results:
        seeds = (f" | 5 seeds {r['seed_mean']:+.4f} +- {r['seed_sd']:.4f} "
                 f"({', '.join(f'{v:+.4f}' for v in r['per_seed'])})" if "per_seed" in r else " | single run")
        print(f"{r['pair']:18s} seed42 {r['diff_seed42']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
              f"p {r['p']:.4f} p_holm {r['p_holm']:.4f}{seeds}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
