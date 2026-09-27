"""Train the sentence classifier head on cached features (Eqs. 5 and 6).

The head is one linear layer from the pooled vector u_j to KEEP/EDIT/DROP
logits. The loss is cross-entropy with class weights w_c = N / (3 N_c) from
the train labels. Features are standardized with train statistics first. That
is an affine map the linear layer could absorb, so the model class is the one
in Eq. 5.

After training, the DROP gate tau_d is tuned on validation (Section 6.2). A
sentence whose argmax is DROP is deleted only if p_DROP >= tau_d. Otherwise it
becomes EDIT.

Usage:
    python -m classifier.train --train features/train --val features/val \
        --eval features/test features/fruit2026 --layer 32 --out heads/layer32
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
    """Features and labels for the rows extract.py has finished."""
    directory = Path(directory)
    rows = json.loads((directory / "progress.json").read_text())["rows"]
    x = np.load(directory / f"layer{layer}.npy", mmap_mode="r")[:rows]
    y = np.load(directory / "labels.npy")[:rows]
    return torch.from_numpy(np.array(x)), torch.from_numpy(y.astype(np.int64))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--eval", nargs="*", default=[], help="feature dirs to report on")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train = load(args.train, args.layer)
    x_val, y_val = load(args.val, args.layer)
    mean, std = (t.to(device) for t in moments(x_train))
    weights = class_weights(y_train).to(device)
    print(f"train {len(y_train)} sentences, val {len(y_val)}, "
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
            loss = F.cross_entropy(head(xb), yb, weight=weights, reduction="sum") / len(idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        head.eval()
        val_f1 = scores(y_val, predict(head, x_val, mean, std, device).argmax(-1))["macro_f1"]
        print(f"epoch {epoch}  loss {total / len(y_train):.4f}  val macro-F1 {val_f1:.4f}")
        if val_f1 > best:
            best, stale = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    head.load_state_dict(best_state)
    head.eval()

    val_probs = predict(head, x_val, mean, std, device)
    sweep = {float(tau): scores(y_val, decide(val_probs, tau))["macro_f1"] for tau in TAU_GRID}
    tau = max(sweep, key=sweep.get)
    print(f"tau_d {tau}  val macro-F1 {sweep[tau]:.4f}")

    metrics = {"layer": args.layer, "tau_d": tau, "tau_sweep": sweep,
               "val": scores(y_val, decide(val_probs, tau))}
    for directory in args.eval:
        x, y = load(directory, args.layer)
        result = scores(y, decide(predict(head, x, mean, std, device), tau))
        metrics[directory] = result
        per_class = "  ".join(f"{name} {result[name]['f1']:.3f}" for name in LABELS)
        print(f"{directory}: macro-F1 {result['macro_f1']:.4f}  {per_class}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
                "layer": args.layer, "tau_d": tau}, out / "head.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
