"""LLaDA-free sentence classifier baseline (team's "why use LLaDA at all"
last-resort idea; also the original working-notes doc's "non-diffusion
encoder" control, Section 4.4). Embeds each sentence -- and, for a fair
comparison against experiments.evidence_train, its article's evidence too --
with a small pretrained encoder that has NO evidence conditioning at all,
then trains the SAME 3-way head as classifier.train. This isolates what
LLaDA's bidirectional, evidence-conditioned representation buys over generic
sentence semantics: if this baseline matches or beats LLaDA's numbers, the
8B backbone isn't earning its cost, at least not for this classifier.

No GPU, no Ada, no cached features needed -- reads the collated JSONL
directly and runs entirely on CPU in minutes. Default encoder is
sentence-transformers/all-MiniLM-L6-v2 (~80MB) via plain transformers
(mean-pooling implemented by hand below, so no new sentence-transformers
package dependency).

CAVEAT: same as experiments.evidence_train -- the team's audit
(audit/noise.json) estimates ~40-60% label noise per class in the training
labels. A weak result here does not prove LLaDA is necessary; it could be
the same label noise dragging every approach down equally.

Without --val, a seeded fraction of the train articles is held out, chosen by
classifier.train.split_by_article, the same rule the LLaDA heads use.

Usage:
    python -m experiments.baseline_train \
        --train ../datasets/collated/train.jsonl --out heads/baseline
    python -m experiments.baseline_train \
        --train ../datasets/collated/train.jsonl --val ../datasets/collated/val.jsonl \
        --eval ../datasets/collated/test.jsonl --out heads/baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from classifier.inputs import LABEL_ID, LABELS, read_jsonl
from classifier.train import TAU_GRID, class_weights, decide, moments, scores, split_by_article

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL, device="cpu"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, texts, batch_size=64):
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            vecs.append(pooled.cpu())
        return torch.cat(vecs)


def load_records(path):
    records = list(read_jsonl(path))
    sentences, sent_labels, sent_article = [], [], []
    evidence_texts, evidence_article = [], []
    for a, record in enumerate(records):
        for sentence, entry in zip(record["source_sentences"], record["labels"]):
            sentences.append(sentence)
            sent_labels.append(LABEL_ID[entry["label"]])
            sent_article.append(a)
        for snippet in record["evidence"]:
            evidence_texts.append(f"{snippet['title']}: {snippet['text']}")
            evidence_article.append(a)
    return (sentences, np.array(sent_labels, dtype=np.int64), np.array(sent_article, dtype=np.int64),
            evidence_texts, np.array(evidence_article, dtype=np.int64), len(records))


def build_features(path, embedder):
    sentences, y, sent_article, evidence_texts, evidence_article, n_articles = load_records(path)

    print(f"  embedding {len(sentences)} sentences...")
    x_sent = embedder.embed(sentences)

    d = x_sent.shape[1]
    evidence_agg = torch.zeros(n_articles, d)
    if evidence_texts:
        print(f"  embedding {len(evidence_texts)} evidence snippets...")
        x_evidence = embedder.embed(evidence_texts)
        for a in range(n_articles):
            mask = evidence_article == a
            if mask.any():
                evidence_agg[a] = x_evidence[mask].mean(0)

    x = torch.cat([x_sent, evidence_agg[sent_article]], dim=1)
    return x, torch.from_numpy(y), sent_article, n_articles


@torch.no_grad()
def predict(head, x, mean, std, device, batch=8192):
    probs = []
    for i in range(0, len(x), batch):
        xb = (x[i:i + batch].to(device).float() - mean) / std
        probs.append(head(xb).softmax(-1).cpu())
    return torch.cat(probs)


def train_one(args, x_train, y_train, x_val, y_val, seed, device, embedder, out, held_rows=None):
    torch.manual_seed(seed)
    mean, std = (t.to(device) for t in moments(x_train))
    weights = class_weights(y_train).to(device)
    print(f"seed {seed}: train {len(y_train)} sentences (feature dim {x_train.shape[1]}), val {len(y_val)}, "
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
    print(f"seed {seed}: tau_d {tau}  val macro-F1 {sweep[tau]:.4f}")

    metrics = {"model": args.model, "tau_d": tau, "tau_sweep": sweep,
               "holdout": None if args.val else args.holdout, "seed": seed,
               "val": scores(y_val, decide(val_probs, tau)), "feature_dim": x_train.shape[1]}
    for path in args.eval:
        x, y, _, _ = build_features(path, embedder)
        result = scores(y, decide(predict(head, x, mean, std, device), tau))
        metrics[path] = result
        per_class = "  ".join(f"{name} {result[name]['f1']:.3f}" for name in LABELS)
        print(f"{path}: macro-F1 {result['macro_f1']:.4f}  {per_class}")

    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
                "model": args.model, "tau_d": tau}, out / "head.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if args.save_predictions and held_rows is not None:
        np.savez_compressed(out / "heldout_predictions.npz", rows=held_rows,
                            probs=val_probs.numpy().astype(np.float32), labels=y_val.numpy().astype(np.int8),
                            tau_d=tau)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", help="collated JSONL for val; if omitted a holdout is carved from train")
    parser.add_argument("--holdout", type=float, default=0.1, help="fraction of train articles held out when --val is not given")
    parser.add_argument("--eval", nargs="*", default=[])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", help="train once per seed into <out>/seed<N>")
    parser.add_argument("--cache", help="npz of train embeddings, written on first use and read after")
    parser.add_argument("--save-predictions", action="store_true",
                        help="write held-out probabilities to heldout_predictions.npz")
    args = parser.parse_args()
    if args.val and args.seeds:
        parser.error("--seeds varies the held-out split, so it needs the train holdout, not --val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = Embedder(args.model, device=str(device))

    print(f"train: {args.train}")
    if args.cache and Path(args.cache).exists():
        cached = np.load(args.cache)
        x_all, y_all = torch.from_numpy(cached["x"]), torch.from_numpy(cached["y"])
        article, n_articles = cached["article"], int(cached["n_articles"])
    else:
        x_all, y_all, article, n_articles = build_features(args.train, embedder)
        if args.cache:
            np.savez(args.cache, x=x_all.numpy(), y=y_all.numpy(), article=article, n_articles=n_articles)
    if args.val:
        print(f"val: {args.val}")
        x_val, y_val, _, _ = build_features(args.val, embedder)
        train_one(args, x_all, y_all, x_val, y_val, args.seed, device, embedder, Path(args.out))
        return
    for seed in args.seeds or [args.seed]:
        fit, held = split_by_article(article, n_articles, args.holdout, seed)
        print(f"seed {seed}: held out {args.holdout:.0%} of train articles ({int(held.sum())} sentences)")
        out = Path(args.out) / f"seed{seed}" if args.seeds else Path(args.out)
        train_one(args, x_all[fit], y_all[fit], x_all[held], y_all[held], seed, device, embedder, out,
                  held.nonzero()[:, 0].numpy())


if __name__ == "__main__":
    main()
