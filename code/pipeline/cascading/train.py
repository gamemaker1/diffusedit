"""Consequence prediction head — Section 6.4.

Architecture (Equation 8):
    g_jk = σ( MLP_ψ( [u_j ; u_k ; u_j ⊙ u_k ; |u_j − u_k|] ) )

where:
  - u_j = pooled LLaDA hidden state of the *corrected* (already repaired) sentence j
  - u_k = pooled LLaDA hidden state of the *outdated* (not yet repaired) sentence k
  - [; ] is concatenation, ⊙ is elementwise product, |·| is elementwise abs-diff
  - MLP has one hidden layer of width 512

Training:
  - Positive pairs: from mine.py (direct edit ↔ cascade edit)
  - Negative pairs: same direct edit + KEEP sentences (3 negatives per positive)
  - Loss: binary cross-entropy
  - Optimizer: AdamW(lr=1e-3, wd=1e-2), batch 512, early stop on val AUROC

Usage:
    python -m cascading.train \\
        --features features/train \\
        --pairs pairs/cascade_pairs.jsonl \\
        --out heads/cascade
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


def load_pairs(pairs_path: str | Path) -> tuple[list[dict], list[dict], list[int]]:
    """Load mined cascade pairs from mine.py output.

    Each line: {"article_id", "j_row", "k_row", "label": 0|1, "similarity"}

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


def pairs_to_tensors(
    pairs: list[dict],
    features: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert pair list → (u_j, u_k, labels) float tensors.

    Pairs reference sentence rows (j_row, k_row) into the features array.
    """
    j_rows = np.array([p["j_row"] for p in pairs], dtype=np.int64)
    k_rows = np.array([p["k_row"] for p in pairs], dtype=np.int64)
    ys     = np.array([p["label"] for p in pairs], dtype=np.float32)

    u_j = torch.from_numpy(np.array(features[j_rows], dtype=np.float32))
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
    """Binary AUROC via the Mann-Whitney U statistic."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(u) / (len(pos) * len(neg))


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

    if best_state is not None:
        head.load_state_dict(best_state)
    return {"best_val_auroc": best_auroc, "history": history}


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
    parser.add_argument("--pairs", required=True,
                        help="JSONL of mined cascade pairs from cascading.mine")
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
    features, article_idx, _ = load_features(args.features, layer=args.layer)
    d = features.shape[1]
    print(f"  {len(features)} sentences, d={d}")

    print(f"Loading pairs from {args.pairs}...")
    all_pairs = load_pairs(args.pairs)
    n_pos = sum(p["label"] for p in all_pairs)
    print(f"  {len(all_pairs)} pairs  ({n_pos} positive, {len(all_pairs)-n_pos} negative)")

    train_pairs, val_pairs = split_pairs_by_article(all_pairs, args.holdout, args.seed)
    print(f"  train: {len(train_pairs)}  val: {len(val_pairs)}")

    train_uj, train_uk, train_y = pairs_to_tensors(train_pairs, features)
    val_uj,   val_uk,   val_y   = pairs_to_tensors(val_pairs, features)

    head = ConsequenceHead(d=d, hidden=args.hidden)
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
        "pairs": str(Path(args.pairs).resolve()),
        "layer": args.layer,
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
