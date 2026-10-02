"""Train the sentence classifier head on cached features (Eqs. 5 and 6).

The head is one linear layer from the pooled vector u_j to KEEP/EDIT/DROP
logits. The loss is cross-entropy with class weights w_c = N / (3 N_c) from
the train labels. Features are standardized with train statistics first. That
is an affine map the linear layer could absorb, so the model class is the one
in Eq. 5.

A seeded 10% of the train articles is held out, split by article so that no
article has sentences on both sides. The held-out part drives early stopping
and tunes the DROP gate tau_d (Section 6.2). A sentence whose argmax is DROP is
deleted only if p_DROP >= tau_d. Otherwise it becomes EDIT.

--loss focal replaces the weighted cross-entropy with the weighted focal
loss -w_y (1 - p_y)^gamma log p_y. --seeds trains once per seed, each with its
own held-out split, initialization and batch order, and loads the features
once.

Usage:
    python -m classifier.train --train features/train --layer 32 --out heads/layer32
    python -m classifier.train --train features/train --layer 24 --loss focal \
        --seeds 42 1 2 3 4 --save-predictions --out heads/focal/layer24
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from classifier.inputs import DROP, EDIT, LABELS

TAU_GRID = np.round(np.arange(0.35, 0.951, 0.05), 2)


def load(directory, layer):
    """Features, labels, article indices and article count for the finished rows."""
    directory = Path(directory)
    progress = json.loads((directory / "progress.json").read_text())
    rows = progress["rows"]
    x = np.load(directory / f"layer{layer}.npy", mmap_mode="r")[:rows]
    y = np.load(directory / "labels.npy")[:rows]
    article = np.load(directory / "article.npy")[:rows]
    return torch.from_numpy(np.array(x)), torch.from_numpy(y.astype(np.int64)), article, progress["articles"]


def split_by_article(article, n_articles, fraction, seed):
    """Row masks for train and held-out.

    A seeded `fraction` of the article indices 0..n_articles-1 is held out, so
    the sentence head and the token head hold out the same articles.
    """
    rng = np.random.default_rng(seed)
    held = rng.choice(n_articles, size=max(1, round(fraction * n_articles)), replace=False)
    holdout = np.isin(article, held)
    return torch.from_numpy(~holdout), torch.from_numpy(holdout)


def moments(x, chunk=65536):
    total = torch.zeros(x.shape[1], dtype=torch.float64)
    squares = torch.zeros_like(total)
    for i in range(0, len(x), chunk):
        part = x[i:i + chunk].double()
        total += part.sum(0)
        squares += (part * part).sum(0)
    mean = total / len(x)
    std = (squares / len(x) - mean * mean).clamp_min(1e-12).sqrt()
    return mean.float(), std.float()


def class_weights(y):
    counts = torch.bincount(y, minlength=len(LABELS)).float()
    return len(y) / (len(LABELS) * counts.clamp_min(1))


@torch.no_grad()
def predict(head, x, mean, std, device, batch=8192):
    probs = []
    for i in range(0, len(x), batch):
        xb = (x[i:i + batch].to(device).float() - mean) / std
        probs.append(head(xb).softmax(-1).cpu())
    return torch.cat(probs)


def decide(probs, tau):
    pred = probs.argmax(-1)
    pred[(pred == DROP) & (probs[:, DROP] < tau)] = EDIT
    return pred


def scores(y, pred):
    out = {}
    for c, name in enumerate(LABELS):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[name] = {"precision": p, "recall": r, "f1": f1, "support": tp + fn}
    out["macro_f1"] = sum(out[name]["f1"] for name in LABELS) / len(LABELS)
    return out


def focal_loss(logits, y, weights, gamma):
    """Class-weighted focal loss, -w_y (1 - p_y)^gamma log p_y, summed.

    gamma = 0 gives the weighted cross-entropy used by default. A larger gamma
    shrinks the loss of sentences the head already classifies with high
    probability, so training concentrates on the hard ones.
    """
    logp = F.log_softmax(logits, -1).gather(1, y[:, None])[:, 0]
    return -(weights[y] * (1 - logp.exp()) ** gamma * logp).sum()


def train_one(args, x_all, y_all, article, n_articles, seed, out, device):
    torch.manual_seed(seed)
    fit, held = split_by_article(article, n_articles, args.holdout, seed)
    x_train, y_train = x_all[fit], y_all[fit]
    x_held, y_held = x_all[held], y_all[held]
    mean, std = (t.to(device) for t in moments(x_train))
    weights = class_weights(y_train).to(device)
    print(f"seed {seed}: train {len(y_train)} sentences, held out {len(y_held)}, loss {args.loss}"
          f"{f' gamma {args.gamma}' if args.loss == 'focal' else ''}, "
          f"weights {dict(zip(LABELS, weights.tolist()))}")

    head = torch.nn.Linear(x_train.shape[1], len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best, best_state, stale = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(y_train))
        total = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xb = (x_train[idx].to(device).float() - mean) / std
            yb = y_train[idx].to(device)
            if args.loss == "focal":
                loss = focal_loss(head(xb), yb, weights, args.gamma) / len(idx)
            else:
                loss = F.cross_entropy(head(xb), yb, weight=weights, reduction="sum") / len(idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        head.eval()
        held_f1 = scores(y_held, predict(head, x_held, mean, std, device).argmax(-1))["macro_f1"]
        print(f"epoch {epoch}  loss {total / len(y_train):.4f}  held-out macro-F1 {held_f1:.4f}")
        if held_f1 > best:
            best, stale = held_f1, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    head.load_state_dict(best_state)
    head.eval()

    held_probs = predict(head, x_held, mean, std, device)
    sweep = {float(tau): scores(y_held, decide(held_probs, tau))["macro_f1"] for tau in TAU_GRID}
    tau = max(sweep, key=sweep.get)
    print(f"seed {seed}: tau_d {tau}  held-out macro-F1 {sweep[tau]:.4f}")

    metrics = {"layer": args.layer, "tau_d": tau, "tau_sweep": sweep, "holdout": args.holdout,
               "seed": seed, "loss": args.loss, "gamma": args.gamma if args.loss == "focal" else None,
               "heldout": scores(y_held, decide(held_probs, tau))}
    for directory in args.eval:
        x, y, _, _ = load(directory, args.layer)
        result = scores(y, decide(predict(head, x, mean, std, device), tau))
        metrics[directory] = result
        per_class = "  ".join(f"{name} {result[name]['f1']:.3f}" for name in LABELS)
        print(f"{directory}: macro-F1 {result['macro_f1']:.4f}  {per_class}")

    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
                "layer": args.layer, "tau_d": tau}, out / "head.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if args.save_predictions:
        np.savez_compressed(out / "heldout_predictions.npz", rows=held.nonzero()[:, 0].numpy(),
                            article=article[held.numpy()], probs=held_probs.numpy().astype(np.float32),
                            labels=y_held.numpy().astype(np.int8), tau_d=tau)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--holdout", type=float, default=0.1, help="fraction of train articles held out")
    parser.add_argument("--eval", nargs="*", default=[], help="feature dirs to report on")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="train once per seed (split, initialization, order) into <out>/seed<N>")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--gamma", type=float, default=2.0, help="focusing exponent of --loss focal")
    parser.add_argument("--save-predictions", action="store_true",
                        help="write held-out probabilities to heldout_predictions.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_all, y_all, article, n_articles = load(args.train, args.layer)
    if args.seeds:
        for seed in args.seeds:
            train_one(args, x_all, y_all, article, n_articles, seed, Path(args.out) / f"seed{seed}", device)
    else:
        train_one(args, x_all, y_all, article, n_articles, args.seed, Path(args.out), device)


if __name__ == "__main__":
    main()
