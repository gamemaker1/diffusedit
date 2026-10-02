"""Train the token staleness scorer on cached features (Eq. 3 / paper Eq. 7).

Same shape as the sentence head (classifier.train) but a single sigmoid
output per token instead of a 3-way softmax per sentence -- only ever run on
EDIT-sentence tokens. Reads classifier.extract's output directly:
tokens/layer{L}.npy and tokens/labels.npy, where L is config.json's
"token_layer" -- there is exactly one token layer per extraction run (no
layer sweep, unlike the sentence classifier's --layer), so it's read from
config.json automatically rather than passed on the command line.

Loss is binary cross-entropy with a positive-class weight for imbalance,
since most tokens even within an EDIT sentence are usually still fine.

The proposal's decision rule (Section 6.2) merges tokens with p_stale >= tau
into contiguous spans (capped at half the sentence) for infilling. tau is
picked here on validation PR-AUC's operating point, not accuracy, since the
positive class is a minority even in this already EDIT-filtered set.

Without --val, a seeded fraction of the train articles is held out, chosen by
classifier.train.split_by_article. With the same seed and holdout fraction,
the sentence head and the token head hold out the same articles. Their ids go
to heldout_ids.json in --out, for infilling.run --ids.

--loss focal replaces the weighted binary cross-entropy with a binary focal
loss. --seeds trains once per seed, each with its own held-out split, and
loads the token features once.

Usage:
    python -m staleness.train --train features/train --out heads/staleness
    python -m staleness.train --train features/train --loss focal --seeds 42 1 2 3 4 \
        --save-predictions --out heads/staleness_focal
    python -m staleness.train --train features/train --val features/val \
        --eval features/test --out heads/staleness
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from classifier.train import split_by_article

TAU_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def token_layer_of(directory):
    config = json.loads((Path(directory) / "config.json").read_text())
    layer = config.get("token_layer")
    if not layer:
        raise SystemExit(f"{directory}: config.json has no token_layer -- was this "
                          f"extracted with --tokens (the default) rather than --no-tokens?")
    return layer


def load(directory, layer):
    """Token features and labels for the rows classifier.extract has
    finished (progress.json's "tokens" count, not "rows" -- that one is the
    sentence count).
    """
    directory = Path(directory)
    n = json.loads((directory / "progress.json").read_text())["tokens"]
    x = np.load(directory / "tokens" / f"layer{layer}.npy", mmap_mode="r")[:n]
    y = np.load(directory / "tokens" / "labels.npy")[:n]
    return torch.from_numpy(np.array(x)), torch.from_numpy(y.astype(np.float32))


class Rows:
    """Rows `idx` of `x`, gathered on access, so a holdout split copies nothing."""

    def __init__(self, x, idx):
        self.x, self.idx = x, idx
        self.shape = (len(idx), x.shape[1])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        return self.x[self.idx[k]]


def token_articles(directory):
    """Article index of every finished token row, and the finished article count."""
    directory = Path(directory)
    progress = json.loads((directory / "progress.json").read_text())
    row = np.load(directory / "tokens" / "row.npy")[:progress["tokens"]]
    return np.load(directory / "article.npy")[row], progress["articles"]


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


def pos_weight(y):
    pos = y.sum().clamp_min(1)
    neg = (len(y) - pos).clamp_min(1)
    return neg / pos


@torch.no_grad()
def predict(head, x, mean, std, device, batch=8192):
    probs = []
    for i in range(0, len(x), batch):
        xb = (x[i:i + batch].to(device).float() - mean) / std
        probs.append(head(xb).sigmoid().cpu().squeeze(-1))
    return torch.cat(probs)


def pr_auc(y, probs):
    """Average precision via step-summation over recall -- no sklearn
    dependency needed for one curve.
    """
    order = torch.argsort(probs, descending=True)
    y_sorted = y[order]
    tp = torch.cumsum(y_sorted, 0)
    fp = torch.cumsum(1 - y_sorted, 0)
    precision = tp / (tp + fp)
    recall = tp / y.sum().clamp_min(1)
    recall_prev = torch.cat([torch.zeros(1), recall[:-1]])
    return ((recall - recall_prev) * precision).sum().item()


def scores(y, pred):
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "support": tp + fn}


def auroc(y, probs):
    """Probability that a random stale token outscores a random correct one."""
    ranks = torch.empty(len(probs), dtype=torch.float64)
    ranks[torch.argsort(probs)] = torch.arange(1, len(probs) + 1, dtype=torch.float64)
    n1 = y.sum().item()
    n0 = len(y) - n1
    return ((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


def focal_bce(logits, y, pos_weight, gamma):
    """Binary focal loss with the stale class weighted by pos_weight, summed.

    -[w y (1 - p)^gamma log p + (1 - y) p^gamma log(1 - p)]. gamma = 0 gives
    the weighted binary cross-entropy used by default.
    """
    logp, log1mp = F.logsigmoid(logits), F.logsigmoid(-logits)
    p = logp.exp()
    return -(pos_weight * y * (1 - p) ** gamma * logp + (1 - y) * p ** gamma * log1mp).sum()


def train_one(args, layer, x_train, y_train, x_val, y_val, seed, out, device, heldout=None):
    torch.manual_seed(seed)
    mean, std = (t.to(device) for t in moments(x_train))
    pw = pos_weight(y_train).to(device)
    print(f"seed {seed}: token_layer {layer}, train {len(y_train)} tokens ({y_train.mean():.3f} stale), "
          f"val {len(y_val)}, pos_weight {pw.item():.2f}, loss {args.loss}"
          f"{f' gamma {args.gamma}' if args.loss == 'focal' else ''}")

    head = torch.nn.Linear(x_train.shape[1], 1).to(device)
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
            logits = head(xb).squeeze(-1)
            if args.loss == "focal":
                loss = focal_bce(logits, yb, pw, args.gamma) / len(idx)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pw, reduction="sum") / len(idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        head.eval()
        val_probs = predict(head, x_val, mean, std, device)
        val_pr_auc = pr_auc(y_val, val_probs)
        print(f"epoch {epoch}  loss {total / len(y_train):.4f}  val PR-AUC {val_pr_auc:.4f}")
        if val_pr_auc > best:
            best, stale = val_pr_auc, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    head.load_state_dict(best_state)
    head.eval()

    val_probs = predict(head, x_val, mean, std, device)
    sweep = {float(tau): scores(y_val, (val_probs >= tau).float())["f1"] for tau in TAU_GRID}
    tau = max(sweep, key=sweep.get)
    val_auroc = auroc(y_val, val_probs)
    print(f"seed {seed}: tau {tau}  val F1 {sweep[tau]:.4f}  val PR-AUC {best:.4f}  val AUROC {val_auroc:.4f}")

    metrics = {"layer": layer, "tau": tau, "tau_sweep": sweep, "val_pr_auc": best, "val_auroc": val_auroc,
               "val": scores(y_val, (val_probs >= tau).float()),
               "holdout": None if args.val else args.holdout, "seed": seed, "loss": args.loss,
               "gamma": args.gamma if args.loss == "focal" else None}
    for directory in args.eval:
        eval_layer = token_layer_of(directory)
        if eval_layer != layer:
            raise SystemExit(f"{directory} token_layer {eval_layer} != train/val token_layer {layer}")
        x, y = load(directory, layer)
        probs = predict(head, x, mean, std, device)
        result = scores(y, (probs >= tau).float())
        result["pr_auc"] = pr_auc(y, probs)
        metrics[directory] = result
        print(f"{directory}: PR-AUC {result['pr_auc']:.4f}  F1 {result['f1']:.4f}  "
              f"P {result['precision']:.3f}  R {result['recall']:.3f}")

    out.mkdir(parents=True, exist_ok=True)
    if heldout is not None:
        (out / "heldout_ids.json").write_text(json.dumps(heldout["ids"]))
        if args.save_predictions:
            np.savez_compressed(out / "heldout_predictions.npz", rows=heldout["rows"],
                                probs=val_probs.numpy().astype(np.float16), labels=y_val.numpy().astype(np.int8),
                                tau=tau)
    torch.save({"state_dict": head.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
                "layer": layer, "tau": tau}, out / "head.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", help="feature dir for val; if omitted a holdout is carved from train")
    parser.add_argument("--holdout", type=float, default=0.1, help="fraction of train articles held out when --val is not given")
    parser.add_argument("--eval", nargs="*", default=[], help="feature dirs to report on")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="train once per seed (held-out split, initialization, order) into <out>/seed<N>")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--gamma", type=float, default=2.0, help="focusing exponent of --loss focal")
    parser.add_argument("--save-predictions", action="store_true",
                        help="write held-out token probabilities to heldout_predictions.npz")
    args = parser.parse_args()
    if args.val and args.seeds:
        parser.error("--seeds varies the held-out split, so it needs the train holdout, not --val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = token_layer_of(args.train)
    if args.val:
        val_layer = token_layer_of(args.val)
        if val_layer != layer:
            raise SystemExit(f"train token_layer {layer} != val token_layer {val_layer}")
        x_train, y_train = load(args.train, layer)
        x_val, y_val = load(args.val, layer)
        train_one(args, layer, x_train, y_train, x_val, y_val, args.seed, Path(args.out), device)
        return

    x_all, y_all = load(args.train, layer)
    article, n_articles = token_articles(args.train)
    ids = json.loads((Path(args.train) / "ids.json").read_text())
    for seed in args.seeds or [args.seed]:
        fit, held = split_by_article(article, n_articles, args.holdout, seed)
        heldout = {"ids": [ids[a] for a in np.unique(article[held.numpy()])],
                   "rows": held.nonzero()[:, 0].numpy()}
        x_train, y_train = Rows(x_all, fit.nonzero()[:, 0]), y_all[fit]
        x_val, y_val = Rows(x_all, held.nonzero()[:, 0]), y_all[held]
        print(f"seed {seed}: held out {args.holdout:.0%} of train articles ({len(y_val)} tokens)")
        out = Path(args.out) / f"seed{seed}" if args.seeds else Path(args.out)
        train_one(args, layer, x_train, y_train, x_val, y_val, seed, out, device, heldout)


if __name__ == "__main__":
    main()
