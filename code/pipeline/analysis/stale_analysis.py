"""Staleness head on held-out tokens: score distributions, threshold curve,
per-sentence masking rules, and per-token scores for example sentences."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/diffusedit/pipeline")
from classifier.train import split_by_article
from classifier.inputs import build, read_jsonl
from staleness.train import token_articles
from transformers import AutoTokenizer

F = Path("/home/diffusedit/features/train")
H = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/diffusedit/heads/staleness/head.pt")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/diffusedit/stale_analysis.json")
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
ckpt = torch.load(H, map_location="cpu")
article, n_articles = token_articles(F)
_, held = split_by_article(article, n_articles, 0.1, SEED)
idx = np.flatnonzero(held.numpy())
x = np.load(F / "tokens" / f"layer{ckpt['layer']}.npy", mmap_mode="r")
y = np.load(F / "tokens" / "labels.npy")[idx].astype(np.int64)
row = np.load(F / "tokens" / "row.npy")[idx]
w, b = ckpt["state_dict"]["weight"][0].double().numpy(), ckpt["state_dict"]["bias"].double().item()
mean, std = ckpt["mean"].double().numpy(), ckpt["std"].double().numpy()
p = np.empty(len(idx))
for i in range(0, len(idx), 50000):
    xb = (np.asarray(x[idx[i:i + 50000]], dtype=np.float64) - mean) / std
    p[i:i + 50000] = 1 / (1 + np.exp(-(xb @ w + b)))
base = y.mean()

bins = np.linspace(0, 1, 51)
hist = {"bins": bins.tolist(), "stale": np.histogram(p[y == 1], bins)[0].tolist(),
        "fresh": np.histogram(p[y == 0], bins)[0].tolist()}
curve = []
for t in np.round(np.arange(0.02, 0.99, 0.02), 2):
    m = p >= t; tp = (m & (y == 1)).sum()
    curve.append({"tau": float(t), "precision": float(tp / max(m.sum(), 1)), "recall": float(tp / y.sum()),
                  "masked": float(m.mean())})

# per-sentence rules
starts = np.flatnonzero(np.r_[True, row[1:] != row[:-1]]); ends = np.r_[starts[1:], len(row)]
def evaluate(select):
    tp = fp = 0
    for a, e in zip(starts, ends):
        m = select(p[a:e], y[a:e]); tp += (m & (y[a:e] == 1)).sum(); fp += (m & (y[a:e] == 0)).sum()
    return {"precision": float(tp / max(tp + fp, 1)), "recall": float(tp / y.sum()), "masked": float((tp + fp) / len(y))}
def topk(scores, k):
    m = np.zeros(len(scores), bool); m[np.argsort(-scores)[:k]] = True; return m
rules = {
    "global tau 0.40 (trained)": evaluate(lambda s, t: s >= 0.40),
    "global tau 0.60 (tau_match)": evaluate(lambda s, t: s >= 0.60),
    "per-sentence top 28.5%": evaluate(lambda s, t: topk(s, int(round(base * len(s))))),
    "per-sentence top-k, k = true stale count (oracle count)": evaluate(lambda s, t: topk(s, int(t.sum()))),
}
# sentence-level ranking quality: AUROC within sentences that have both classes
aucs = []
for a, e in zip(starts, ends):
    t, s = y[a:e], p[a:e]
    if 0 < t.sum() < len(t):
        r = s.argsort().argsort() + 1; n1 = t.sum(); n0 = len(t) - n1
        aucs.append((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
all_auc = None
order = p.argsort(); rk = np.empty(len(p)); rk[order] = np.arange(1, len(p) + 1)
n1 = y.sum(); n0 = len(y) - n1; all_auc = float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
stale_per_sentence = np.array([y[a:e].mean() for a, e in zip(starts, ends)])

# examples: four held-out sentences with 15-40 tokens and some stale tokens
rng = np.random.default_rng(3)
cands = [k for k, (a, e) in enumerate(zip(starts, ends)) if 15 <= e - a <= 40 and 0 < y[a:e].sum() < e - a]
pick = rng.choice(cands, 4, replace=False)
art_of_row = np.load(F / "article.npy"); ids = json.loads((F / "ids.json").read_text())
first_row = np.searchsorted(art_of_row, np.arange(art_of_row.max() + 1))
want = {}
for k in pick:
    r = int(row[starts[k]]); a = int(art_of_row[r]); want[ids[a]] = want.get(ids[a], []) + [(k, r - int(first_row[a]))]
tok = AutoTokenizer.from_pretrained("/home/diffusedit/models/LLaDA-8B-Base", trust_remote_code=True)
examples = []
for rec in read_jsonl("/home/diffusedit/data/train.jsonl"):
    if rec["id"] not in want: continue
    ex = build(rec, tok)
    for k, j in want[rec["id"]]:
        a0, a1 = ex.spans[j]; s, e = starts[k], ends[k]
        assert a1 - a0 == e - s and (ex.stale[j] == y[s:e]).all()
        tgt = rec["target_sentences"][rec["labels"][j]["target_idx"]]["text"]
        examples.append({"tokens": [tok.decode([int(t)]) for t in ex.input_ids[a0:a1]],
                         "scores": p[s:e].tolist(), "stale": y[s:e].tolist(), "target": tgt})
out = {"base_rate": float(base), "tokens": int(len(y)), "sentences": int(len(starts)), "hist": hist, "curve": curve,
       "rules": rules, "auc_global": all_auc, "auc_within_sentence_mean": float(np.mean(aucs)),
       "auc_sentences": len(aucs), "stale_fraction_quantiles": np.quantile(stale_per_sentence, [.1, .25, .5, .75, .9]).tolist(),
       "examples": examples}
OUT.write_text(json.dumps(out))
print(json.dumps({k: v for k, v in out.items() if k not in ("hist", "curve", "examples")}, indent=1))
