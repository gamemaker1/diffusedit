"""Sentence classifier augmented with evidence features (team's Approach A).

Same 3-way KEEP/EDIT/DROP head as classifier.train, but each sentence's
feature vector is [pooled sentence vector ; mean-pooled evidence vector for
that article] instead of just the sentence vector alone -- testing whether
giving the classifier explicit access to evidence (beyond whatever the
sentence's own hidden state already picked up via attention) helps.
Evidence is mean-pooled across all of an article's snippets rather than kept
as separate per-snippet features, since the snippet count varies per
article and there's no way yet to know which snippet is most relevant --
exactly the "let the model learn" approach the team discussed.

Needs experiments.evidence_extract's output in addition to
classifier.extract's normal output (same articles, same order, same split).

CAVEAT: the team's audit (audit/noise.json) estimates ~40-60% label noise
per class in the training data. If this doesn't beat classifier.train's
numbers, that is NOT strong evidence evidence-augmentation doesn't help --
label noise may be swamping any real signal either way. Compare against
classifier.train run on the SAME split for an apples-to-apples number.

Usage:
    python -m experiments.evidence_train \
        --train features/train --train-evidence evidence_features/train \
        --val features/val --val-evidence evidence_features/val \
        --eval features/test=evidence_features/test \
        --layer 32 --out heads/evidence_layer32
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from classifier.inputs import LABELS
from classifier.train import TAU_GRID, class_weights, decide, moments, scores, split_by_article


def load_sentence(directory, layer):
    directory = Path(directory)
    rows = json.loads((directory / "progress.json").read_text())["rows"]
    x = np.load(directory / f"layer{layer}.npy", mmap_mode="r")[:rows]
    y = np.load(directory / "labels.npy")[:rows]
    article = np.load(directory / "article.npy")[:rows]
    return torch.from_numpy(np.array(x)), torch.from_numpy(y.astype(np.int64)), article


def load_evidence_per_article(directory, layer, n_articles):
    """Mean-pools each article's evidence-snippet vectors into one vector.

    Articles with zero evidence rows get the dataset-wide mean evidence
    vector (rare edge case, printed not silenced) -- a zero vector would
    look like a real, meaningful "no evidence" signal rather than "we have
    nothing better to guess here".
    """
    directory = Path(directory)
    rows = json.loads((directory / "progress.json").read_text())["rows"]
    x = np.array(np.load(directory / f"layer{layer}.npy", mmap_mode="r")[:rows]).astype(np.float32)
    article = np.load(directory / "article.npy")[:rows]

    d = x.shape[1]
    agg = np.zeros((n_articles, d), dtype=np.float32)
    counts = np.zeros(n_articles, dtype=np.int64)
    for a in range(n_articles):
        mask = article == a
        if mask.any():
            agg[a] = x[mask].mean(0)
            counts[a] = mask.sum()

    missing = int((counts == 0).sum())
    if missing:
        agg[counts == 0] = x.mean(0)
        print(f"  {missing}/{n_articles} articles have no evidence rows -- using the "
              f"dataset-wide mean evidence vector for those")
    return torch.from_numpy(agg)


def build_features(sentence_dir, evidence_dir, layer):
    x_sent, y, sent_article = load_sentence(sentence_dir, layer)
    n_articles = int(sent_article.max()) + 1
    evidence_agg = load_evidence_per_article(evidence_dir, layer, n_articles)
    x_evidence = evidence_agg[sent_article]
    return torch.cat([x_sent, x_evidence], dim=1), y, sent_article


@torch.no_grad()
def predict(head, x, mean, std, device, batch=8192):
    probs = []
    for i in range(0, len(x), batch):
        xb = (x[i:i + batch].to(device).float() - mean) / std
        probs.append(head(xb).softmax(-1).cpu())
    return torch.cat(probs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--train-evidence", required=True)
    parser.add_argument("--val", default=None, help="sentence feature dir for val; if omitted a holdout is carved from train")
    parser.add_argument("--val-evidence", default=None, help="evidence feature dir for val; required when --val is given")
    parser.add_argument("--holdout", type=float, default=0.1, help="fraction of train articles held out when --val is not given")
    parser.add_argument("--eval", nargs="*", default=[], help="pairs like features/test=evidence_features/test")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading train...")
    x_all, y_all, train_article = build_features(args.train, args.train_evidence, args.layer)

    if args.val and args.val_evidence:
        print("loading val...")
        x_val, y_val, _ = build_features(args.val, args.val_evidence, args.layer)
        x_train, y_train = x_all, y_all
    else:
        n_articles = int(train_article.max()) + 1
        fit, held = split_by_article(train_article, n_articles, args.holdout, args.seed)
        x_train, y_train = x_all[fit], y_all[fit]
        x_val, y_val = x_all[held], y_all[held]
        print(f"  no --val given: held out {args.holdout:.0%} of train articles ({len(y_val)} sentences)")
    del x_all

    mean, std = (t.to(device) for t in moments(x_train))
    weights = class_weights(y_train).to(device)
    print(f"train {len(y_train)} sentences (feature dim {x_train.shape[1]}), val {len(y_val)}, "
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
               "val": scores(y_val, decide(val_probs, tau)), "feature_dim": x_train.shape[1]}
    for pair in args.eval:
        feat_dir, ev_dir = pair.split("=")
        x, y, _ = build_features(feat_dir, ev_dir, args.layer)
        result = scores(y, decide(predict(head, x, mean, std, device), tau))
        metrics[pair] = result
        per_class = "  ".join(f"{name} {result[name]['f1']:.3f}" for name in LABELS)
        print(f"{pair}: macro-F1 {result['macro_f1']:.4f}  {per_class}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
                "layer": args.layer, "tau_d": tau}, out / "head.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
