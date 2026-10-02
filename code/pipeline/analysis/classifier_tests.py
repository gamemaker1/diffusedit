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

Usage:
    python -m analysis.classifier_tests --sweep outputs/jl/heads/sweep --out outputs/jl/heads/sweep/tests.json
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
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--out")
    args = parser.parse_args()
    sweep = Path(args.sweep)
    configs = {"baseline": sweep / "baseline", "ce24": sweep / "cls-ce/layer24", "ce32": sweep / "cls-ce/layer32",
               "focal24": sweep / "cls-focal/layer24", "focal32": sweep / "cls-focal/layer32"}
    pairs = [("ce24", "baseline"), ("ce24", "ce32"), ("ce24", "focal24"), ("ce32", "focal32")]
    rng = np.random.default_rng(0)

    article = articles_of(configs["ce24"] / "seed42/heldout_predictions.npz")
    preds = {}
    for name, path in configs.items():
        rows, pred, y = load(path / "seed42/heldout_predictions.npz")
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
        per_seed = []
        for s in SEEDS:
            _, fa, ya = load(configs[a_name] / f"seed{s}/heldout_predictions.npz")
            _, fb, _ = load(configs[b_name] / f"seed{s}/heldout_predictions.npz")
            per_seed.append(macro_f1(ya, fa) - macro_f1(ya, fb))
        results.append({"pair": f"{a_name}:{b_name}", "diff_seed42": observed, "ci_low": float(low),
                        "ci_high": float(high), "p": float(p), "per_seed": per_seed,
                        "seed_mean": float(np.mean(per_seed)), "seed_sd": float(np.std(per_seed, ddof=1))})
    for r, adjusted in zip(results, holm(np.array([r["p"] for r in results]))):
        r["p_holm"] = float(adjusted)
    for r in results:
        print(f"{r['pair']:18s} seed42 {r['diff_seed42']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
              f"p {r['p']:.4f} p_holm {r['p_holm']:.4f} | 5 seeds {r['seed_mean']:+.4f} +- {r['seed_sd']:.4f} "
              f"({', '.join(f'{v:+.4f}' for v in r['per_seed'])})")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
