"""Per-token scores, labels and a 2-D view of held-out token states for the staleness head."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/diffusedit/pipeline")
from classifier.train import split_by_article
from staleness.train import token_articles

F = Path("/home/diffusedit/features/train")
ckpt = torch.load("/home/diffusedit/heads/staleness/head.pt", map_location="cpu")
article, n_articles = token_articles(F)
_, held = split_by_article(article, n_articles, 0.1, 42)
idx = np.flatnonzero(held.numpy())
x = np.load(F / "tokens" / f"layer{ckpt['layer']}.npy", mmap_mode="r")
y_all = np.load(F / "tokens" / "labels.npy")
w = ckpt["state_dict"]["weight"][0].double().numpy(); b = ckpt["state_dict"]["bias"].double().item()
mean, std = ckpt["mean"].double().numpy(), ckpt["std"].double().numpy()

# logits for every held-out token, for the binned curve
logit = np.empty(len(idx))
for i in range(0, len(idx), 50000):
    logit[i:i + 50000] = ((np.asarray(x[idx[i:i + 50000]], np.float64) - mean) / std) @ w + b
y = y_all[idx].astype(np.int64)
p = 1 / (1 + np.exp(-logit))
bins = np.linspace(0, 1, 41)
which = np.clip(np.digitize(p, bins) - 1, 0, 39)
binned = [{"lo": float(bins[k]), "hi": float(bins[k + 1]), "n": int((which == k).sum()),
           "stale": int(y[which == k].sum())} for k in range(40)]

# 20k-token sample: 2-D view, x = head logit, y = top PC of states with the head direction removed
rng = np.random.default_rng(0)
s = np.sort(rng.choice(len(idx), 20000, replace=False))
z = (np.asarray(x[idx[s]], np.float64) - mean) / std
u = w / np.linalg.norm(w)
r = z - np.outer(z @ u, u)
r -= r.mean(0)
_, sv, vt = np.linalg.svd(r[:5000], full_matrices=False)
pc = r @ vt[:2].T
var = (sv ** 2 / (sv ** 2).sum())[:2]
# best single direction in the residual: logistic-free LDA direction on the sample
mu1, mu0 = r[y[s] == 1].mean(0), r[y[s] == 0].mean(0)
lda = np.linalg.lstsq(np.cov(r[:5000].T) + 1e-1 * np.eye(r.shape[1]), mu1 - mu0, rcond=None)[0]
lda_proj = r @ (lda / np.linalg.norm(lda))
out = {"binned": binned, "sample": {"logit": logit[s].tolist(), "score": p[s].tolist(), "label": y[s].tolist(),
       "pc1": pc[:, 0].tolist(), "pc2": pc[:, 1].tolist(), "lda_resid": lda_proj.tolist()},
       "pc_var_share": var.tolist(), "tau": float(ckpt["tau"])}
Path("/home/diffusedit/stale_scatter.json").write_text(json.dumps(out))
print("ok", len(s), var)
