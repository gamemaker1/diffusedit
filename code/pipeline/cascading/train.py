"""Consequence prediction head — Section 6.4.

Architecture (Equation 8):
    g_jk = σ( MLP_ψ( [u_j ; u_k ; u_j ⊙ u_k ; |u_j − u_k|] ) )

where:
  - u_j = pooled LLaDA hidden state of the *corrected* (already repaired) sentence j
  - u_k = pooled LLaDA hidden state of the *outdated* (not yet repaired) sentence k
  - [; ] is concatenation, ⊙ is elementwise product, |·| is elementwise abs-diff
  - MLP has one hidden layer of width 512

Training:
  - Inputs are standardized with the moments of the training vectors, which
    the head stores as buffers, so callers pass raw pooled states
  - Positive pairs: the cascade links of cascading.links (direct edit ↔ cascade edit)
  - Negative pairs: same direct edit + KEEP sentences (3 negatives per positive)
  - Loss: binary cross-entropy
  - Optimizer: AdamW(lr=1e-3, wd=1e-2), batch 512, early stop on val AUROC
  - tau_c is the threshold with the best F1 on the validation pairs, saved
    in config.json, the default of the consequence detector

u_j comes from --features-after, the features of the oracle-repaired view
(cascading.views), so that the direct edit enters in its corrected version.
Without --features-after it comes from --features, the outdated version. Pairs
name sentences by article and sentence index ("article_id", "j", "k").

Usage:
    python -m cascading.train --features features/cascade/source --features-after features/cascade/after \\
        --pairs ../datasets/synthetic/cascade/pairs.jsonl --out heads/consequence
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConsequenceHead(nn.Module):
    """Pair-feature MLP for consequence prediction (Eq. 8).

    Input: two sentence pooled vectors u_j (corrected) and u_k (candidate).
    The four-block feature [u_j ; u_k ; u_j⊙u_k ; |u_j−u_k|] has size 4*d.
    """

    def __init__(self, d: int = 4096, hidden: int = 512):
        super().__init__()
        self.d = d
        self.hidden = hidden
        # Standardization with the moments of the training vectors, saved with the head.
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("std", torch.ones(d))
        self.mlp = nn.Sequential(
            nn.Linear(4 * d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, u_j: torch.Tensor, u_k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_j: [B, d] pooled vectors of repaired sentences
            u_k: [B, d] pooled vectors of candidate sentences
        Returns:
            [B, 1] raw logit (apply sigmoid for probability)
        """
        u_j, u_k = (u_j - self.mean) / self.std, (u_k - self.mean) / self.std
        feat = torch.cat([u_j, u_k, u_j * u_k, (u_j - u_k).abs()], dim=-1)
        return self.mlp(feat)

    def predict(self, u_j: torch.Tensor, u_k: torch.Tensor) -> torch.Tensor:
        """Convenience: returns sigmoid probability [B] in inference_mode."""
        with torch.inference_mode():
            return torch.sigmoid(self(u_j, u_k)).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(directory: str | Path, layer: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load sentence-level pooled features from classifier.extract output.

    Returns:
        features: float16 [n_sents, d]
        article:  int32  [n_sents]   — article index per sentence
        labels:   int8   [n_sents]   — KEEP=0, EDIT=1, DROP=2
    """
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if layer is None:
        layers = config.get("layers", [32])
        layer = max(layers)

    n_rows = json.loads((directory / "progress.json").read_text())["rows"]
    features = np.load(directory / f"layer{layer}.npy", mmap_mode="r")[:n_rows]
    article  = np.load(directory / "article.npy")[:n_rows]
    labels   = np.load(directory / "labels.npy")[:n_rows]
    return features, article, labels


def load_pairs(pairs_path: str | Path) -> list[dict]:
    """Load cascade pairs from cascading.links output.

    Each line: {"article_id", "j", "k", "label": 0|1}

    Returns:
        pairs: list of pair dicts
    """
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def first_rows(directory: str | Path) -> dict[str, int]:
    """Feature row of the first sentence of every article in a classifier.extract directory."""
    directory = Path(directory)
    ids = json.loads((directory / "ids.json").read_text())
    article = np.load(directory / "article.npy")
    starts = np.flatnonzero(np.r_[True, article[1:] != article[:-1]])
    return {ids[article[r]]: int(r) for r in starts}


def resolve(pairs: list[dict], starts: dict[str, int]) -> list[dict]:
    """Add the feature rows j_row and k_row, skipping pairs of articles without features."""
    return [{**p, "j_row": starts[p["article_id"]] + p["j"], "k_row": starts[p["article_id"]] + p["k"]}
            for p in pairs if p["article_id"] in starts]


def pairs_to_tensors(
    pairs: list[dict],
    features: np.ndarray,
    features_after: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert pair list → (u_j, u_k, labels) float tensors.

    Pairs reference sentence rows (j_row, k_row). u_j is read from
    features_after when given, and u_k always from features.
    """
    j_rows = np.array([p["j_row"] for p in pairs], dtype=np.int64)
    k_rows = np.array([p["k_row"] for p in pairs], dtype=np.int64)
    ys     = np.array([p["label"] for p in pairs], dtype=np.float32)

    source_j = features if features_after is None else features_after
    u_j = torch.from_numpy(np.array(source_j[j_rows], dtype=np.float32))
    u_k = torch.from_numpy(np.array(features[k_rows], dtype=np.float32))
    y   = torch.from_numpy(ys)
    return u_j, u_k, y


def split_pairs_by_article(
    pairs: list[dict],
    holdout_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Article-level train/val split (same strategy as classifier.train)."""
    rng = random.Random(seed)
    article_ids = list({p["article_id"] for p in pairs})
    rng.shuffle(article_ids)
    n_val = max(1, int(len(article_ids) * holdout_fraction))
    val_ids = set(article_ids[:n_val])
    train_pairs = [p for p in pairs if p["article_id"] not in val_ids]
    val_pairs   = [p for p in pairs if p["article_id"] in val_ids]
    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary AUROC via the Mann-Whitney U statistic, with tied scores given their mean rank."""
    n1 = int((labels == 1).sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    ends = np.r_[starts[1:], len(scores)]
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = (a + b + 1) / 2
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """The threshold of a 0.05 grid with the highest F1 on (scores, labels), and that F1."""
    best = (0.5, -1.0)
    for tau in np.round(np.arange(0.05, 0.951, 0.05), 2):
        pred = scores >= tau
        tp = int((pred & (labels == 1)).sum())
        p = tp / max(int(pred.sum()), 1)
        r = tp / max(int((labels == 1).sum()), 1)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best[1]:
            best = (float(tau), f1)
    return best


def train(
    head: ConsequenceHead,
    train_u_j: torch.Tensor,
    train_u_k: torch.Tensor,
    train_y: torch.Tensor,
    val_u_j: torch.Tensor,
    val_u_k: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    lr: float = 1e-3,
    wd: float = 1e-2,
    batch: int = 512,
    max_epochs: int = 30,
    patience: int = 3,
) -> dict:
    """Train the consequence head. Returns best metrics dict."""
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    # Positive class weight: n_neg / n_pos
    n_pos = train_y.sum().item()
    n_neg = len(train_y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    train_ds = TensorDataset(train_u_j, train_u_k, train_y)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True)

    best_auroc, best_state, patience_left = -1.0, None, patience
    history = []

    for epoch in range(1, max_epochs + 1):
        head.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()
        for uj, uk, y in loader:
            uj, uk, y = uj.to(device), uk.to(device), y.to(device)
            logits = head(uj, uk).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        head.eval()
        with torch.inference_mode():
            val_logits = head(val_u_j.to(device), val_u_k.to(device)).squeeze(-1)
            val_loss = F.binary_cross_entropy_with_logits(
                val_logits, val_y.to(device), pos_weight=pos_weight
            ).item()
        val_scores = torch.sigmoid(val_logits).cpu().numpy()
        val_labels = val_y.numpy()
        val_auc = auroc(val_scores, val_labels)

        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(n_batches, 1), 4),
            "val_loss": round(val_loss, 4),
            "val_auroc": round(val_auc, 4) if not np.isnan(val_auc) else None,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(f"  epoch {epoch:2d}  train_loss {row['train_loss']:.4f}  "
              f"val_loss {row['val_loss']:.4f}  val_auroc {row['val_auroc']}  "
              f"({row['seconds']:.1f}s)")

        if val_auc > best_auroc:
            best_auroc = val_auc
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stop at epoch {epoch} (best val AUROC {best_auroc:.4f})")
                break

    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        val_scores = torch.sigmoid(head(val_u_j.to(device), val_u_k.to(device)).squeeze(-1)).cpu().numpy()
    tau_c, val_f1 = best_threshold(val_scores, val_y.numpy())
    print(f"  tau_c {tau_c}  val F1 {val_f1:.4f}")
    return {"best_val_auroc": best_auroc, "tau_c": tau_c, "val_f1": val_f1, "history": history}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--features", required=True,
                        help="feature directory from classifier.extract")
    parser.add_argument("--features-after",
                        help="features of the oracle-repaired view, for u_j (cascading.views)")
    parser.add_argument("--pairs", required=True,
                        help="JSONL of cascade pairs from cascading.links")
    parser.add_argument("--out", required=True,
                        help="output directory for head.pt and config.json")
    parser.add_argument("--layer", type=int, default=None,
                        help="which extracted layer to use (default: deepest)")
    parser.add_argument("--hidden", type=int, default=512,
                        help="MLP hidden width (default: 512, per paper)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--holdout", type=float, default=0.1,
                        help="article-level holdout fraction for early stopping")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading features from {args.features}...")
    layer = args.layer or max(json.loads((Path(args.features) / "config.json").read_text())["layers"])
    features, article_idx, _ = load_features(args.features, layer=layer)
    d = features.shape[1]
    print(f"  {len(features)} sentences, d={d}")

    features_after = None
    if args.features_after:
        if json.loads((Path(args.features_after) / "ids.json").read_text()) != \
                json.loads((Path(args.features) / "ids.json").read_text()):
            raise SystemExit("--features and --features-after hold different articles")
        features_after, _, _ = load_features(args.features_after, layer=layer)

    print(f"Loading pairs from {args.pairs}...")
    all_pairs = resolve(load_pairs(args.pairs), first_rows(args.features))
    n_pos = sum(p["label"] for p in all_pairs)
    print(f"  {len(all_pairs)} pairs  ({n_pos} positive, {len(all_pairs)-n_pos} negative)")

    train_pairs, val_pairs = split_pairs_by_article(all_pairs, args.holdout, args.seed)
    print(f"  train: {len(train_pairs)}  val: {len(val_pairs)}")

    train_uj, train_uk, train_y = pairs_to_tensors(train_pairs, features, features_after)
    val_uj,   val_uk,   val_y   = pairs_to_tensors(val_pairs, features, features_after)

    if len(set(val_y.tolist())) < 2 or len(set(train_y.tolist())) < 2:
        raise SystemExit("the train or validation pairs hold a single class; AUROC and early stopping need both")
    head = ConsequenceHead(d=d, hidden=args.hidden)
    both = torch.cat([train_uj, train_uk])
    head.mean.copy_(both.mean(0))
    head.std.copy_(both.std(0).clamp_min(1e-6))
    print(f"\nTraining ConsequenceHead  d={d}  hidden={args.hidden}  device={device}")
    metrics = train(
        head, train_uj, train_uk, train_y, val_uj, val_uk, val_y,
        device=device, lr=args.lr, wd=args.wd, batch=args.batch,
        max_epochs=args.max_epochs, patience=args.patience,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "d": d,
        "hidden": args.hidden,
        "features": str(Path(args.features).resolve()),
        "features_after": str(Path(args.features_after).resolve()) if args.features_after else None,
        "pairs": str(Path(args.pairs).resolve()),
        "layer": layer,
        "seed": args.seed,
        **metrics,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(
        {"state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, **config},
        out / "head.pt",
    )
    print(f"\nSaved to {out}/  best val AUROC: {metrics['best_val_auroc']:.4f}")


if __name__ == "__main__":
    main()
