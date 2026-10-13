"""Precision, recall and masked fraction of a trained staleness head per tau.

staleness.train tunes tau for token F1, which favours recall: at its tau the
head masks about 2.3 tokens per stale token. This script scores the same
held-out articles (same seed and holdout fraction) at every tau of a grid and
picks tau_match, the tau whose share of masked tokens is closest to the share
of stale tokens. It also reports tau_f05, the tau with the best F0.5, which
weighs precision twice as much as recall.

Usage:
    python -m staleness.sweep --train features/train --head heads/staleness/head.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from classifier.train import split_by_article
from staleness.train import Rows, load, predict, token_articles

GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", help="default sweep.json next to the head")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.head, map_location=device)
    head = torch.nn.Linear(ckpt["state_dict"]["weight"].shape[1], 1).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    x_all, y_all = load(args.train, ckpt["layer"])
    article, n_articles = token_articles(args.train)
    _, held = split_by_article(article, n_articles, args.holdout, args.seed)
    x, y = Rows(x_all, held.nonzero()[:, 0]), y_all[held]
    probs = predict(head, x, ckpt["mean"].to(device), ckpt["std"].to(device), device)
    stale = y.mean().item()

    rows = []
    for tau in GRID:
        pred = probs >= tau
        tp = (pred & (y == 1)).sum().item()
        p = tp / max(pred.sum().item(), 1)
        r = tp / max((y == 1).sum().item(), 1)
        f = lambda beta: (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0
        rows.append({"tau": float(tau), "precision": p, "recall": r, "f1": f(1), "f05": f(0.5),
                     "masked_fraction": pred.float().mean().item()})
        print(f"tau {tau:.2f}  P {p:.3f}  R {r:.3f}  F1 {f(1):.3f}  F0.5 {f(0.5):.3f}  "
              f"masked {rows[-1]['masked_fraction']:.3f}")
    tau_match = min(rows, key=lambda row: abs(row["masked_fraction"] - stale))["tau"]
    tau_f05 = max(rows, key=lambda row: row["f05"])["tau"]
    print(f"stale fraction {stale:.3f}, tau_match {tau_match}, tau_f05 {tau_f05}, trained tau {ckpt['tau']}")

    out = Path(args.out) if args.out else Path(args.head).with_name("sweep.json")
    out.write_text(json.dumps({"stale_fraction": stale, "tau_match": tau_match, "tau_f05": tau_f05,
                               "trained_tau": ckpt["tau"], "grid": rows}, indent=2))


if __name__ == "__main__":
    main()
