"""Shortcut probes: how well do surface features predict the labels?"""
import json, sys, random, collections, re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
L = {"KEEP": 0, "EDIT": 1, "DROP": 2}
def norm(s): return " ".join(s.split())
def load(path):
    rows = [json.loads(l) for l in open(path)]
    text, y, art, feats = [], [], [], []
    bad = 0
    for a, r in enumerate(rows):
        assert len(r["labels"]) == len(r["source_sentences"])
        ev = [norm(e["text"]) for e in r["evidence"]]
        titles = {e["title"] for e in r["evidence"]}
        for s, l in zip(r["source_sentences"], r["labels"]):
            for x, z in l["stale_spans"]:
                bad += not (0 <= x < z <= len(s))
            if l["label"] == "EDIT": bad += not l["stale_spans"] or s == l["target"]
            if l["label"] == "KEEP": bad += s != l["target"]
            n = norm(s)[:250]
            hit = [e for e in ev if n in e]
            feats.append([len(ev), bool(hit), any(e != norm(s) and "'" in e for e in hit), len(hit),
                          any(e == norm(s) for e in ev), len(titles), "Correction Notice" in titles,
                          "General Reference Document" in titles])
            text.append(s); y.append(L[l["label"]]); art.append(a)
    return rows, text, np.array(y), np.array(art), np.array(feats, dtype=float), bad
for path in sys.argv[1:]:
    rows, text, y, art, feats, bad = load(path)
    rng = np.random.default_rng(42); held = np.isin(art, rng.choice(len(rows), len(rows) // 10, replace=False))
    print(f"== {path}: {len(rows)} articles, {len(y)} sentences, labels {np.bincount(y).tolist()}, invalid {bad}")
    X = HashingVectorizer(n_features=2**20, ngram_range=(1, 2), alternate_sign=False).transform(text)
    Fs = csr_matrix(StandardScaler().fit_transform(feats))
    for name, M in (("sentence words only", X), ("evidence-match flags only", Fs), ("both", hstack([X, Fs]).tocsr())):
        clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=None, class_weight="balanced", random_state=0).fit(M[~held], y[~held])
        p = clf.predict(M[held])
        f = f1_score(y[held], p, average=None)
        print(f"  {name:26s} macro-F1 {f.mean():.3f}  KEEP {f[0]:.3f} EDIT {f[1]:.3f} DROP {f[2]:.3f}")
    edit = [(s, l) for r in rows for s, l in zip(r["source_sentences"], r["labels"]) if l["label"] == "EDIT"]
    stale = sum(b - a for s, l in edit for a, b in l["stale_spans"]); chars = sum(len(s) for s, _ in edit)
    top = collections.Counter(s[a:b] for s, l in edit for a, b in l["stale_spans"])
    print(f"  stale share of EDIT characters {stale / chars:.3f}, distinct stale strings {len(top)}, top 12 cover {sum(v for _, v in top.most_common(12)) / sum(top.values()):.3f}")
    ne = [len(r["evidence"]) for r in rows]; ec = [sum(len(e["text"]) + len(e["title"]) + 3 for e in r["evidence"]) for r in rows]
    print(f"  snippets per article {np.mean(ne):.2f}, evidence chars mean {np.mean(ec):.0f}, p99 {np.percentile(ec, 99):.0f}, over 2400 chars {np.mean(np.array(ec) > 2400):.3f}")
