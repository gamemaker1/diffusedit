"""Token staleness head with learned attention over the evidence tokens.

The linear head of staleness.train scores each token from its own hidden
state. On the synthetic set it keeps AUROC 0.976 when the evidence is removed
from the input, so it detects tokens that are unlikely in context and makes
little use of the evidence. This head compares each token with the evidence
explicitly. For an EDIT-sentence token with hidden state h and the evidence
token states e_1..e_m of its article, all from the token layer:

    q = W_q h,  k_i = W_k e_i,  v_i = W_v e_i          (H heads, width r)
    c = attention(q, [k_null; k_1..k_m], [v_null; v_1..v_m])
    t = W_t h
    p_stale = sigmoid(MLP([t; c; t * c; |t - c|]))

The learned null slot (k_null, v_null) lets a token attend to nothing when
the evidence does not mention it. The four-block comparison is the one of the
consequence head (Eq. 8). Hidden states are standardized with the moments of
the fit tokens and of the fit evidence tokens.

Training uses weighted binary cross-entropy, early stopping on held-out
PR-AUC, and the article holdout of classifier.train.split_by_article, so with
the same seed and fraction it holds out the same articles as staleness.train.
Batches are whole articles, so each token attends only to its own evidence.

Two controls score the held-out tokens with altered evidence:

    shuffled   each article reads the evidence of another held-out article
    none       every article reads no evidence, only the null slot

A head that reads the evidence loses AUROC under both. A head that ignores it
scores the same.

Needs classifier.extract output with tokens/evidence_layer{L}.npy (the
default with evidence in the input).

Output directory:
    head.pt           state dict, configuration, moments, layer, tau, tau_match
    metrics.json      held-out PR-AUC, AUROC, F1 at tau, and the two controls
    sweep.json        precision, recall and masked share per tau, and tau_match,
                      the tau whose masked share is closest to the stale share
    heldout_ids.json  held-out article ids, for infilling.run --ids

Usage:
    python -m staleness.attention --train features/train --out heads/staleness_attention
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from classifier.train import split_by_article
from staleness.train import (TAU_GRID, Rows, auroc, load, moments, pos_weight, pr_auc, scores, token_articles,
                             token_layer_of)


class EvidenceAttention(torch.nn.Module):
    def __init__(self, d, dim=256, heads=4, hidden=256, dropout=0.1):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} is not a multiple of heads {heads}")
        self.heads = heads
        self.query = torch.nn.Linear(d, dim)
        self.key = torch.nn.Linear(d, dim)
        self.value = torch.nn.Linear(d, dim)
        self.token = torch.nn.Linear(d, dim)
        self.null_key = torch.nn.Parameter(torch.zeros(dim))
        self.null_value = torch.nn.Parameter(torch.zeros(dim))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * dim, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))

    def split(self, x):
        a, n, dim = x.shape
        return x.view(a, n, self.heads, dim // self.heads).transpose(1, 2)

    def forward(self, tokens, evidence, evidence_mask, weights=False):
        """Logits [A, T] for padded tokens [A, T, d] and evidence [A, M, d].

        evidence_mask [A, M] is True at real evidence tokens. With weights=True
        the attention weights [A, H, T, M + 1] are returned too, the null slot
        first.
        """
        a = tokens.shape[0]
        keys = torch.cat([self.null_key.expand(a, 1, -1), self.key(evidence)], 1)
        values = torch.cat([self.null_value.expand(a, 1, -1), self.value(evidence)], 1)
        mask = torch.cat([evidence_mask.new_ones(a, 1), evidence_mask], 1)
        q, k, v = self.split(self.query(tokens)), self.split(keys), self.split(values)
        logits = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
        attn = logits.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        context = (attn @ v).transpose(1, 2).reshape(a, tokens.shape[1], -1)
        t = self.token(tokens)
        out = self.mlp(torch.cat([t, context, t * context, (t - context).abs()], -1)).squeeze(-1)
        return (out, attn) if weights else out


class Articles:
    """Token and evidence-token states of one classifier.extract directory, by article."""

    def __init__(self, directory, layer):
        directory = Path(directory)
        self.x, self.y = load(directory, layer)
        self.article, self.n_articles = token_articles(directory)
        path = directory / "tokens" / f"evidence_layer{layer}.npy"
        if not path.exists():
            raise SystemExit(f"{path} is missing. Extract with evidence and --evidence-tokens (the defaults).")
        self.evidence = np.load(path, mmap_mode="r")
        self.offsets = np.load(directory / "tokens" / "evidence_offsets.npy")
        starts = np.searchsorted(self.article, np.arange(self.n_articles + 1))
        self.tokens = {a: (int(starts[a]), int(starts[a + 1])) for a in range(self.n_articles) if starts[a + 1] > starts[a]}

    def evidence_of(self, a):
        return torch.from_numpy(np.array(self.evidence[self.offsets[a]:self.offsets[a + 1]]))


def evidence_moments(data, articles, limit=200_000, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.concatenate([np.arange(data.offsets[a], data.offsets[a + 1]) for a in articles])
    rows = np.sort(rng.choice(rows, min(limit, len(rows)), replace=False))
    return moments(torch.from_numpy(np.array(data.evidence[rows])))


def batches(articles, data, batch_tokens, rng=None):
    """Lists of articles holding about batch_tokens EDIT tokens each."""
    order = list(articles)
    if rng is not None:
        rng.shuffle(order)
    batch, n = [], 0
    for a in order:
        batch.append(a)
        n += data.tokens[a][1] - data.tokens[a][0]
        if n >= batch_tokens:
            yield batch
            batch, n = [], 0
    if batch:
        yield batch


def collate(data, batch, norm, device, evidence_from=None, no_evidence=False):
    """Padded tokens [A, T, d], token mask, evidence [A, M, d], evidence mask, labels [n]."""
    mean, std, ev_mean, ev_std = norm
    toks = [data.x[slice(*data.tokens[a])] for a in batch]
    labels = torch.cat([data.y[slice(*data.tokens[a])] for a in batch]).to(device)
    sources = [evidence_from.get(a, a) if evidence_from else a for a in batch]
    evs = [torch.zeros(0, data.x.shape[1]) if no_evidence else data.evidence_of(s) for s in sources]
    t_max, m_max = max(len(t) for t in toks), max(1, max(len(e) for e in evs))
    tokens = torch.zeros(len(batch), t_max, data.x.shape[1], device=device)
    token_mask = torch.zeros(len(batch), t_max, dtype=torch.bool, device=device)
    evidence = torch.zeros(len(batch), m_max, data.x.shape[1], device=device)
    evidence_mask = torch.zeros(len(batch), m_max, dtype=torch.bool, device=device)
    for i, (t, e) in enumerate(zip(toks, evs)):
        tokens[i, :len(t)] = (t.to(device).float() - mean) / std
        token_mask[i, :len(t)] = True
        if len(e):
            evidence[i, :len(e)] = (e.to(device).float() - ev_mean) / ev_std
            evidence_mask[i, :len(e)] = True
    return tokens, token_mask, evidence, evidence_mask, labels


@torch.no_grad()
def predict(head, data, articles, norm, device, batch_tokens, **kw):
    """Probabilities and labels of every token of `articles`, in article order."""
    head.eval()
    probs, ys = [], []
    for batch in batches(articles, data, batch_tokens):
        tokens, token_mask, evidence, evidence_mask, labels = collate(data, batch, norm, device, **kw)
        probs.append(head(tokens, evidence, evidence_mask).sigmoid()[token_mask].cpu())
        ys.append(labels.cpu())
    return torch.cat(probs), torch.cat(ys)


def sweep(y, probs):
    """Precision, recall, F1 and masked share per tau, and the tau whose masked share is closest to the stale share."""
    stale = y.mean().item()
    grid = {}
    for tau in TAU_GRID:
        pred = (probs >= tau).float()
        grid[float(tau)] = {**scores(y, pred), "masked": pred.mean().item()}
    tau_match = min(grid, key=lambda t: abs(grid[t]["masked"] - stale))
    return grid, tau_match, stale


class Scorer:
    """A saved attention head applied to the hidden states of one frozen pass, for infilling.run."""

    def __init__(self, ckpt, device):
        self.head = EvidenceAttention(**ckpt["config"]).to(device)
        self.head.load_state_dict(ckpt["state_dict"])
        self.head.eval()
        self.norm = [ckpt[k].to(device) for k in ("mean", "std", "ev_mean", "ev_std")]

    @torch.no_grad()
    def __call__(self, hidden, token_range, evidence_range):
        """Stale probability of every position in token_range of hidden [L, d]."""
        mean, std, ev_mean, ev_std = self.norm
        tokens = ((hidden[slice(*token_range)].float() - mean) / std)[None]
        evidence = ((hidden[slice(*evidence_range)].float() - ev_mean) / ev_std)[None] if evidence_range \
            else hidden.new_zeros(1, 1, hidden.shape[1]).float()
        mask = torch.ones(evidence.shape[:2], dtype=torch.bool, device=hidden.device)
        if not evidence_range:
            mask[:] = False
        return self.head(tokens, evidence, mask).sigmoid()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout", type=float, default=0.1, help="fraction of train articles held out")
    parser.add_argument("--dim", type=int, default=256, help="attention width r")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256, help="width of the MLP hidden layer")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-tokens", type=int, default=2048, help="EDIT tokens per batch, whole articles")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    layer = token_layer_of(args.train)
    data = Articles(args.train, layer)
    fit_mask, held_mask = split_by_article(data.article, data.n_articles, args.holdout, args.seed)
    fit = sorted({int(a) for a in data.article[fit_mask.numpy()]})
    held = sorted({int(a) for a in data.article[held_mask.numpy()]})
    fit_rows = fit_mask.nonzero()[:, 0]
    mean, std = moments(Rows(data.x, fit_rows))
    ev_mean, ev_std = evidence_moments(data, fit)
    norm = [t.to(device) for t in (mean, std, ev_mean, ev_std)]
    pw = pos_weight(data.y[fit_rows]).to(device)
    d = data.x.shape[1]
    print(f"token_layer {layer}, {len(fit)} fit articles ({len(fit_rows)} tokens), {len(held)} held out, "
          f"{len(data.evidence)} evidence tokens, pos_weight {pw.item():.2f}", flush=True)

    config = {"d": d, "dim": args.dim, "heads": args.heads, "hidden": args.hidden, "dropout": args.dropout}
    head = EvidenceAttention(**config).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best, best_state, stale = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        total, n, t0 = 0.0, 0, time.time()
        for batch in batches(fit, data, args.batch_tokens, rng):
            tokens, token_mask, evidence, evidence_mask, labels = collate(data, batch, norm, device)
            logits = head(tokens, evidence, evidence_mask)[token_mask]
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
            n += len(labels)
        probs, y = predict(head, data, held, norm, device, args.batch_tokens)
        score = pr_auc(y, probs)
        print(f"epoch {epoch}  loss {total / n:.4f}  val PR-AUC {score:.4f}  {time.time() - t0:.0f}s", flush=True)
        if score > best:
            best, stale = score, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    head.load_state_dict(best_state)

    probs, y = predict(head, data, held, norm, device, args.batch_tokens)
    grid, tau_match, stale_share = sweep(y, probs)
    tau = max(grid, key=lambda t: grid[t]["f1"])
    shuffled = dict(zip(held, np.roll(np.random.default_rng(args.seed).permutation(held), 1).tolist()))
    p_shuffled, _ = predict(head, data, held, norm, device, args.batch_tokens, evidence_from=shuffled)
    p_none, _ = predict(head, data, held, norm, device, args.batch_tokens, no_evidence=True)
    metrics = {
        "layer": layer, "tau": tau, "tau_match": tau_match, "val_pr_auc": pr_auc(y, probs),
        "val_auroc": auroc(y, probs), "val": scores(y, (probs >= tau).float()),
        "controls": {
            "shuffled_evidence": {"pr_auc": pr_auc(y, p_shuffled), "auroc": auroc(y, p_shuffled)},
            "no_evidence": {"pr_auc": pr_auc(y, p_none), "auroc": auroc(y, p_none)},
        },
        "holdout": args.holdout, "seed": args.seed, "config": config, "epochs_run": epoch,
    }
    print(f"tau {tau}  F1 {metrics['val']['f1']:.4f}  PR-AUC {metrics['val_pr_auc']:.4f}  "
          f"AUROC {metrics['val_auroc']:.4f}  tau_match {tau_match}")
    for name, c in metrics["controls"].items():
        print(f"control {name}: PR-AUC {c['pr_auc']:.4f}  AUROC {c['auroc']:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = json.loads((Path(args.train) / "ids.json").read_text())
    (out / "heldout_ids.json").write_text(json.dumps([ids[a] for a in held]))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "sweep.json").write_text(json.dumps({"tau_match": tau_match, "stale_share": stale_share,
                                                "grid": grid}, indent=2))
    torch.save({"kind": "attention", "config": config, "state_dict": head.state_dict(),
                "mean": mean, "std": std, "ev_mean": ev_mean, "ev_std": ev_std,
                "layer": layer, "tau": tau, "tau_match": tau_match}, out / "head.pt")


if __name__ == "__main__":
    main()
