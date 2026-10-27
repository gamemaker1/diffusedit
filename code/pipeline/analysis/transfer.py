"""Apply the heads of one train set to the held-out features of another.

Each set is name=<feature dir>:<heads dir>. For every data set B and every head
set A, the block-24 and block-32 sentence heads and the staleness head of A
(seed 42) score the held-out 10% of B's articles, with A's standardization and
thresholds. A = B reproduces the held-out scores of training. The staleness
rows also hold the per-sentence rules that need the sentence of each token:
the top-k rule with the true stale count, and the mean AUROC within sentences.

Usage:
    python -m analysis.transfer --sets fruit=features/train:heads \\
        synthetic=features/synthetic:synthetic/heads --out transfer.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from classifier.train import decide, load, predict, scores, split_by_article
from staleness.train import auroc, pr_auc, token_articles
from staleness.train import predict as predict_tokens


def head_of(path, outputs, device):
    ckpt = torch.load(path, map_location="cpu")
    head = torch.nn.Linear(ckpt["state_dict"]["weight"].shape[1], outputs)
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device).eval(), ckpt["mean"].to(device), ckpt["std"].to(device), ckpt


def sentence_rules(p, y, row):
    """Precision of masking the k highest scores per sentence, k the true stale count, and AUROC within sentences."""
    starts = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
    ends = np.r_[starts[1:], len(row)]
    hit = total = 0
    aucs = []
    for a, e in zip(starts, ends):
        t, s = y[a:e], p[a:e]
        k = int(t.sum())
        hit += int(t[np.argsort(-s)[:k]].sum())
        total += k
        if 0 < k < len(t):
            r = s.argsort().argsort() + 1
            aucs.append((r[t == 1].sum() - k * (k + 1) / 2) / (k * (len(t) - k)))
    return {"sentences": len(starts), "top_k_true_count_precision": hit / max(total, 1),
            "auroc_within_sentence": float(np.mean(aucs)), "sentences_with_both": len(aucs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", required=True, help="name=<feature dir>:<heads dir>")
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = {name: tuple(Path(p) for p in rest.split(":")) for name, rest in (s.split("=") for s in args.sets)}
    report = {}

    for data, (features, _) in sets.items():
        for layer in (24, 32):
            x, y, article, n_articles = load(features, layer)
            _, held = split_by_article(article, n_articles, args.holdout, args.seed)
            x, y = x[held], y[held]
            for source, (_, heads) in sets.items():
                head, mean, std, ckpt = head_of(heads / f"classifier/layer{layer}/head.pt", 3, device)
                probs = predict(head, x, mean, std, device)
                result = scores(y, decide(probs, ckpt["tau_d"]))
                report[f"classifier{layer}/{source}->{data}"] = result
                print(f"classifier {layer} {source} -> {data}: macro-F1 {result['macro_f1']:.4f}  "
                      + "  ".join(f"{c} {result[c]['f1']:.3f}" for c in ("KEEP", "EDIT", "DROP")), flush=True)

        token_article, n_articles = token_articles(features)
        _, held = split_by_article(token_article, n_articles, args.holdout, args.seed)
        idx = np.flatnonzero(held.numpy())
        progress = json.loads((features / "progress.json").read_text())
        layer = json.loads((features / "config.json").read_text())["token_layer"]
        x = torch.from_numpy(np.load(features / "tokens" / f"layer{layer}.npy", mmap_mode="r")[idx])
        y = np.load(features / "tokens" / "labels.npy")[:progress["tokens"]][idx].astype(np.int64)
        row = np.load(features / "tokens" / "row.npy")[:progress["tokens"]][idx]
        for source, (_, heads) in sets.items():
            head, mean, std, ckpt = head_of(heads / "staleness/head.pt", 1, device)
            probs = predict_tokens(head, x, mean, std, device)
            p, target = probs.numpy().astype(np.float64), torch.from_numpy(y).float()
            mask = p >= ckpt["tau"]
            tp = int((mask & (y == 1)).sum())
            result = {"tokens": len(y), "stale_share": float(y.mean()), "tau": ckpt["tau"],
                      "auroc": auroc(target, probs), "pr_auc": pr_auc(target, probs),
                      "precision": tp / max(int(mask.sum()), 1), "recall": tp / int(y.sum()),
                      "masked": float(mask.mean()), **sentence_rules(p, y, row)}
            report[f"staleness/{source}->{data}"] = result
            print(f"staleness {source} -> {data}: AUROC {result['auroc']:.4f}  PR-AUC {result['pr_auc']:.4f}  "
                  f"P {result['precision']:.3f}  R {result['recall']:.3f}  masked {result['masked']:.3f}  "
                  f"top-k {result['top_k_true_count_precision']:.3f}  within {result['auroc_within_sentence']:.3f}",
                  flush=True)
        del x

    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
