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

--evidence-pool chooses how an article's snippets become one vector.

    mean       the mean of all the article's snippet vectors
    nearest    the one snippet with the highest cosine similarity to the
               sentence vector, a different snippet per sentence
    attention  a learned query from the sentence vector attends over the
               article's snippets (keys are a learned projection, values the
               standardized snippet vectors). The attention and the linear
               head train together. Held-out split only, no --val.

Usage:
    python -m experiments.evidence_train --train features/train --layer 24 \
        --evidence-pool attention --out heads/evidence_attention_layer24
    python -m experiments.evidence_train \
        --train features/train \
        --val features/val \
        --eval features/test \
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
    rows = json.loads((directory / "progress.json").read_text())["evidence_rows"]
    x = np.array(np.load(directory / f"evidence_layer{layer}.npy", mmap_mode="r")[:rows]).astype(np.float32)
    article = np.load(directory / "evidence_article.npy")[:rows]

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


def load_evidence_rows(directory, layer):
    """Per-snippet evidence vectors and the article index of each."""
    directory = Path(directory)
    rows = json.loads((directory / "progress.json").read_text())["evidence_rows"]
    x = np.load(directory / f"evidence_layer{layer}.npy", mmap_mode="r")[:rows]
    return torch.from_numpy(np.array(x)), np.load(directory / "evidence_article.npy")[:rows]


def nearest_evidence(x_sent, sent_article, x_ev, ev_article):
    """For each sentence, its article's snippet with the highest cosine similarity.

    Sentences of articles without snippets get the dataset-wide mean snippet.
    """
    out = x_ev.float().mean(0).expand(len(x_sent), -1).clone()
    ev_starts = np.searchsorted(ev_article, np.arange(int(sent_article.max()) + 2))
    sent_starts = np.searchsorted(sent_article, np.arange(int(sent_article.max()) + 2))
    unit = lambda t: F.normalize(t.float(), dim=-1)
    for a in range(len(ev_starts) - 1):
        e0, e1, s0, s1 = ev_starts[a], ev_starts[a + 1], sent_starts[a], sent_starts[a + 1]
        if e1 > e0 and s1 > s0:
            best = (unit(x_sent[s0:s1]) @ unit(x_ev[e0:e1]).T).argmax(1)
            out[s0:s1] = x_ev[e0:e1][best].float()
    return out.half()


def build_features(directory, layer, pool="mean"):
    x_sent, y, sent_article = load_sentence(directory, layer)
    n_articles = int(sent_article.max()) + 1
    if pool == "nearest":
        x_ev, ev_article = load_evidence_rows(directory, layer)
        x_evidence = nearest_evidence(x_sent, sent_article, x_ev, ev_article)
    else:
        x_evidence = load_evidence_per_article(directory, layer, n_articles)[sent_article]
    return torch.cat([x_sent, x_evidence.to(x_sent.dtype)], dim=1), y, sent_article


class AttentionHead(torch.nn.Module):
    """Linear KEEP/EDIT/DROP head on [u ; sum_k a_k e_k], with a_k from a learned query."""

    def __init__(self, d, hidden=256):
        super().__init__()
        self.q = torch.nn.Linear(d, hidden)
        self.k = torch.nn.Linear(d, hidden)
        self.out = torch.nn.Linear(2 * d, len(LABELS))
        self.scale = hidden ** -0.5

    def forward(self, u, e, mask):
        logits = (self.k(e) @ self.q(u)[:, :, None])[..., 0] * self.scale
        a = logits.masked_fill(~mask, float("-inf")).softmax(-1)
        return self.out(torch.cat([u, (a[..., None] * e).sum(1)], -1)), a


def snippet_index(ev_article, n_articles, cap):
    """[n_articles, K] rows of each article's snippets, padded with -1, at most `cap`."""
    counts = np.bincount(ev_article, minlength=n_articles)
    k = int(min(cap, counts.max()))
    index = np.full((n_articles, k), -1, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for a in np.flatnonzero(counts):
        n = min(counts[a], k)
        index[a, :n] = np.arange(starts[a], starts[a] + n)
    return torch.from_numpy(index), int((counts > k).sum())


def train_attention(args, device):
    x_sent, y, article = load_sentence(args.train, args.layer)
    x_ev, ev_article = load_evidence_rows(args.train, args.layer)
    n_articles = int(article.max()) + 1
    index, truncated = snippet_index(ev_article, n_articles, args.max_snippets)
    fit, held = split_by_article(article, n_articles, args.holdout, args.seed)
    fit_rows, held_rows = fit.nonzero()[:, 0], held.nonzero()[:, 0]
    s_mean, s_std = (t.to(device) for t in moments(x_sent[fit_rows]))
    e_mean, e_std = (t.to(device) for t in moments(x_ev))
    art = torch.from_numpy(article).long()
    weights = class_weights(y[fit_rows]).to(device)
    print(f"train {len(fit_rows)} sentences, held out {len(held_rows)}, up to {index.shape[1]} snippets "
          f"per article ({truncated} articles truncated)")

    def inputs(rows):
        u = (x_sent[rows].to(device).float() - s_mean) / s_std
        idx = index[art[rows]]
        mask = (idx >= 0).to(device)
        e = (x_ev[idx.clamp_min(0)].to(device).float() - e_mean) / e_std * mask[..., None]
        mask[~mask.any(1), 0] = True
        return u, e, mask

    @torch.no_grad()
    def probs_of(model, rows):
        out, peak = [], []
        for i in range(0, len(rows), args.batch):
            logits, a = model(*inputs(rows[i:i + args.batch]))
            out.append(logits.softmax(-1).cpu())
            peak.append(a.max(-1).values.cpu())
        return torch.cat(out), torch.cat(peak)

    model = AttentionHead(x_sent.shape[1], args.attention_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, stale = -1.0, None, 0
    y_held = y[held_rows]
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = fit_rows[torch.randperm(len(fit_rows))]
        total = 0.0
        for i in range(0, len(order), args.batch):
            rows = order[i:i + args.batch]
            logits, _ = model(*inputs(rows))
            loss = F.cross_entropy(logits, y[rows].to(device), weight=weights, reduction="sum") / len(rows)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(rows)
        model.eval()
        held_f1 = scores(y_held, probs_of(model, held_rows)[0].argmax(-1))["macro_f1"]
        print(f"epoch {epoch}  loss {total / len(fit_rows):.4f}  held-out macro-F1 {held_f1:.4f}", flush=True)
        if held_f1 > best:
            best, stale = held_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    held_probs, peak = probs_of(model, held_rows)
    sweep = {float(tau): scores(y_held, decide(held_probs, tau))["macro_f1"] for tau in TAU_GRID}
    tau = max(sweep, key=sweep.get)
    print(f"tau_d {tau}  held-out macro-F1 {sweep[tau]:.4f}  mean peak attention {peak.mean():.3f}")
    metrics = {"layer": args.layer, "evidence_pool": "attention", "tau_d": tau, "tau_sweep": sweep,
               "holdout": args.holdout, "seed": args.seed, "val": scores(y_held, decide(held_probs, tau)),
               "mean_peak_attention": peak.mean().item(), "max_snippets": index.shape[1]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "sentence_mean": s_mean.cpu(), "sentence_std": s_std.cpu(),
                "evidence_mean": e_mean.cpu(), "evidence_std": e_std.cpu(), "layer": args.layer,
                "tau_d": tau}, out / "head.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


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
    parser.add_argument("--val", default=None, help="feature dir for val; if omitted a holdout is carved from train")
    parser.add_argument("--holdout", type=float, default=0.1, help="fraction of train articles held out when --val is not given")
    parser.add_argument("--eval", nargs="*", default=[], help="feature dirs to evaluate on")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evidence-pool", choices=["mean", "nearest", "attention"], default="mean")
    parser.add_argument("--attention-hidden", type=int, default=256, help="query/key width for --evidence-pool attention")
    parser.add_argument("--max-snippets", type=int, default=32, help="snippets per article for --evidence-pool attention")
    args = parser.parse_args()
    if args.evidence_pool == "attention" and (args.val or args.eval):
        parser.error("--evidence-pool attention supports the held-out split only")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.evidence_pool == "attention":
        return train_attention(args, device)

    print("loading train...")
    x_all, y_all, train_article = build_features(args.train, args.layer, args.evidence_pool)

    if args.val:
        print("loading val...")
        x_val, y_val, _ = build_features(args.val, args.layer, args.evidence_pool)
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

    metrics = {"layer": args.layer, "evidence_pool": args.evidence_pool, "tau_d": tau, "tau_sweep": sweep,
               "val": scores(y_val, decide(val_probs, tau)), "feature_dim": x_train.shape[1]}
    for directory in args.eval:
        x, y, _ = build_features(directory, args.layer, args.evidence_pool)
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
