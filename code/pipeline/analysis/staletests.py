"""Paired tests between token staleness heads on the same test tokens.

For a pair (A, B) the statistic is PR-AUC(A) - PR-AUC(B) over the EDIT-sentence
tokens of the test split.

    attention:linear     attention head against the linear head (stale-bce, seed 42)
    attention:shuffled   attention head with each article's evidence against
                         the evidence of another test article
    attention:none       attention head with and without the evidence
    linear:noev          linear head on states with and without evidence in the
                         input (stale-bce, seed 42, both)

The unit of resampling is the test article, since tokens of one article are
not independent.

    95% CI   paired bootstrap over articles, --resamples draws
    p        paired permutation test that swaps the two systems' scores inside
             each article, two-sided, --resamples draws
    p_holm   Holm correction over the pairs

PR-AUC is average precision with tied scores taken as one threshold, as in
sklearn's average_precision_score. The probabilities are stored as float16,
so ties are common, and the value can differ in the third or fourth decimal
from the pr_auc of metrics.json, which orders tied tokens arbitrarily.

Every file has to hold the same labels in the same token order (and the same
token rows and article ids where it stores them). A file that does not is
reported and left out, and so is a pair whose file is missing. The article id
of each token comes from the files. A staleness.train file written before it
stored them is matched to --features-test through tokens/row.npy and
article.npy.

Output: a JSON list of {pair, metric, a, b, diff, ci_low, ci_high, p, p_holm,
tokens, articles}.

Usage:
    python -m analysis.staletests --full full --features-test features/full/test \\
        --out full/analysis/stale_tests.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from infilling.significance import holm

PAIRS = (("attention", "linear"), ("attention", "shuffled"), ("attention", "none"), ("linear", "noev"))


class Paired:
    """Scores of A and B on the same n tokens, sorted together once.

    The average precision of any mix of the two under any token weights is then
    a pass over the 2n sorted scores, without sorting again.
    """

    def __init__(self, a, b, y):
        s = np.concatenate([a, b]).astype(np.float32)
        order = np.argsort(-s, kind="stable")
        self.n = len(y)
        self.source = order % self.n
        self.from_b = order >= self.n
        self.y = np.concatenate([y, y])[order].astype(np.float64)
        ranked = s[order]
        self.ends = np.flatnonzero(np.r_[ranked[1:] != ranked[:-1], True])

    def ap(self, wa, wb):
        """Average precision of the scores of A weighted by wa and of B weighted by wb (per token)."""
        w = np.where(self.from_b, wb[self.source], wa[self.source])
        tp = np.cumsum(w * self.y)[self.ends]
        seen = np.cumsum(w)[self.ends]
        if tp[-1] <= 0:
            return np.nan
        precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        return float((np.diff(tp, prepend=0.0) * precision).sum() / tp[-1])


def test(a, b, y, group, rng, resamples):
    """PR-AUC of A and B, their difference, bootstrap 95% interval and permutation p."""
    pair = Paired(a, b, y)
    ones, zeros = np.ones(len(y)), np.zeros(len(y))
    pa, pb = pair.ap(ones, zeros), pair.ap(zeros, ones)
    observed = pa - pb
    n_groups = group.max() + 1
    boot, perm = np.empty(resamples), np.empty(resamples)
    for i in range(resamples):
        w = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)[group].astype(np.float64)
        boot[i] = pair.ap(w, zeros) - pair.ap(zeros, w)
        swap = (rng.random(n_groups) < 0.5)[group].astype(np.float64)
        perm[i] = pair.ap(1 - swap, swap) - pair.ap(swap, 1 - swap)
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    p = (np.sum(np.abs(perm) >= abs(observed) - 1e-12) + 1) / (resamples + 1)
    return pa, pb, observed, float(low), float(high), float(p)


def token_articles(directory):
    """Article index and label of every finished token row of a classifier.extract directory."""
    directory = Path(directory)
    n = json.loads((directory / "progress.json").read_text())["tokens"]
    row = np.load(directory / "tokens/row.npy")[:n]
    return np.load(directory / "article.npy")[row], np.load(directory / "tokens/labels.npy")[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", required=True, help="output directory of jobs/jl/run_full.sh")
    parser.add_argument("--features-test", help="test feature directory, for article ids the files do not store")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--out")
    args = parser.parse_args()
    full = Path(args.full)
    paths = {"linear": full / "heads/sweep/stale-bce/seed42/test_predictions.npz",
             "attention": full / "heads/staleness_attention/test_predictions.npz",
             "noev": full / "noev/heads/sweep/stale-bce/seed42/test_predictions.npz"}
    files = {}
    for name, path in paths.items():
        if path.exists():
            files[name] = dict(np.load(path))
        else:
            print(f"{name}: {path} is missing")
    if not files:
        raise SystemExit("no prediction files")

    # The reference token order is that of the first file. A file that differs is left out.
    ref_name = next(iter(files))
    y = files[ref_name]["labels"].astype(np.int64)
    for name in list(files)[1:]:
        if not np.array_equal(files[name]["labels"].astype(np.int64), y):
            print(f"{name}: labels differ from {ref_name} ({len(files[name]['labels'])} against {len(y)} tokens), left out")
            del files[name]
    for field in ("rows", "article"):
        have = [n for n in files if field in files[n]]
        for name in have[1:]:
            if name in files and not np.array_equal(files[name][field], files[have[0]][field]):
                print(f"{name}: {field} differs from {have[0]}, left out")
                del files[name]

    article = next((files[n]["article"] for n in files if "article" in files[n]), None)
    if article is None:
        if not args.features_test:
            raise SystemExit("no file stores article ids; pass --features-test")
        article, labels = token_articles(args.features_test)
        if not np.array_equal(labels.astype(np.int64), y):
            raise SystemExit(f"{args.features_test}: token labels differ from the prediction files")
        print(f"article ids from {args.features_test}")
    _, group = np.unique(article, return_inverse=True)

    systems = {}
    for name in ("linear", "noev"):
        if name in files:
            systems[name] = files[name]["probs"]
    if "attention" in files:
        systems["attention"] = files["attention"]["probs"]
        systems["shuffled"] = files["attention"]["probs_shuffled_evidence"]
        systems["none"] = files["attention"]["probs_no_evidence"]

    rng = np.random.default_rng(0)
    results = []
    for a_name, b_name in PAIRS:
        if a_name not in systems or b_name not in systems:
            print(f"{a_name}:{b_name}: skipped, a system is missing")
            continue
        pa, pb, diff, low, high, p = test(systems[a_name], systems[b_name], y, group, rng, args.resamples)
        results.append({"pair": f"{a_name}:{b_name}", "metric": "pr_auc", "a": pa, "b": pb, "diff": diff,
                        "ci_low": low, "ci_high": high, "p": p, "tokens": len(y), "articles": int(group.max() + 1)})
    for r, adjusted in zip(results, holm(np.array([r["p"] for r in results]))):
        r["p_holm"] = float(adjusted)
    print(f"{len(y)} test tokens in {group.max() + 1} articles, {args.resamples} resamples")
    for r in results:
        print(f"{r['pair']:20s} PR-AUC {r['a']:.4f} - {r['b']:.4f} = {r['diff']:+.4f} "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p {r['p']:.4f} p_holm {r['p_holm']:.4f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
