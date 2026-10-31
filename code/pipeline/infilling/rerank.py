"""Rerank the logged candidate pools of an infilling run with a classifier
fit on the pools of a val run.

The reranker of infilling.score ranks candidates by 0.5 PLL + 0.5 Ent, less
the edit penalty and the rejection. On the val articles that chose the v2
setting, the chosen candidate scores 0.913 ROUGE-L and the best candidate of
the pool 0.958, so the ranking loses 0.045 of the 0.076 that the pool offers
over copy. Every logged candidate is a whole sentence, so a second ranking
over the logged pools needs no model pass.

Each candidate gets features from its logged scores and from its text against
the source sentence and the evidence. The scores are PLL, Ent, their
differences from the best of the decision, and their ranks. The length
features are the ratio of the fill length to the span length and the length
rule the fill came from. The text features are the word edit distance to the
source, the ROUGE-L against the source, the counts of capitalised words and
numbers the candidate adds, split by whether the evidence holds them, the
count of such words it drops, and the mean and minimum sampler confidence.
The unchanged source sentence enters every pool as one more candidate, so the
reranker can leave a sentence as it is.

A gradient-boosted tree classifier is fit to predict whether a candidate has
the best ROUGE-L of its pool, and the candidate with the highest predicted
probability is the output. The classifier is fit on every pool of the val run
and applied once to the test run. Under cross-validation over val articles
the reranked pools score 0.933 ROUGE-L against 0.913 for the logged choice.

A sentence with two spans was decided span by span, and the candidates of the
second decision were sampled after the first choice, so the reranker chooses
among the logged whole sentences of both decisions and cannot resample.

Usage:
    python -m infilling.rerank fit runs/repair-val2-v2 ../datasets/synthetic/split/val.jsonl \\
        runs/rerank.pkl
    python -m infilling.rerank apply runs/rerank.pkl runs/fix-tuned \\
        ../datasets/synthetic/split/test.jsonl runs/fix-rerank
    python -m infilling.evaluate runs/fix-rerank ../datasets/synthetic/split/test.jsonl
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from sklearn.ensemble import HistGradientBoostingClassifier

from classifier.inputs import read_jsonl
from infilling.evaluate import gold
from infilling.score import PIECE, levenshtein

FEATURES = (
    "pll", "ent", "pll_gap", "ent_gap", "pll_rank", "ent_rank", "length_ratio", "zero", "half", "same",
    "plus2", "copied", "source", "rejected", "edit", "edit_ratio", "confidence_mean", "confidence_min",
    "added_names", "added_supported", "added_unsupported", "dropped_names", "source_rougeL",
    "decision", "decisions", "candidates", "combined", "combined_gap", "copied_words",
)
REJECTED = -50.0


def wordset(text):
    return set(re.findall(r"\w+", text.lower()))


def words(text):
    """Lower-cased words of `text`, with hyphens and apostrophes kept inside a word."""
    return [w.lower() for w in PIECE.findall(text)]


def names(text):
    """Capitalised words and numbers of `text`, lower-cased, as whole pieces."""
    return {w.lower() for w in PIECE.findall(text) if w[0].isupper() or w[0].isdigit()}


def pool(result, record, j, decisions, spans, scorer):
    """Feature rows, texts and ROUGE-L targets of every candidate of sentence j, plus the source."""
    source, target = result["source"][j], gold(record, j)
    evidence = " ".join(f"{s['title']} {s['text']}" for s in record["evidence"])
    evidence_words, evidence_names = wordset(evidence), names(evidence)
    source_words, source_names = words(source), names(source)
    rows, texts = [], []
    for d, decision in enumerate(decisions):
        candidates = decision["candidates"]
        plls = np.array([c["pll"] for c in candidates])
        ents = np.array([c["ent"] for c in candidates])
        combined = 0.5 * plls + 0.5 * ents
        span = (spans[d][1] - spans[d][0]) if d < len(spans) else max(c["lengths"][0] for c in candidates)
        for k, c in enumerate(candidates):
            text = c["text"]
            text_words, text_names = words(text), names(text)
            added = text_names - source_names
            confidence = c["confidence"] or [1.0]
            copied = c.get("copied")
            n = c["lengths"][0]
            edit = levenshtein(text_words, source_words)
            rows.append([
                c["pll"], c["ent"], c["pll"] - plls.max(), c["ent"] - ents.max(),
                (plls > c["pll"]).sum(), (ents > c["ent"]).sum(),
                n / max(span, 1), float(n == 0), float(n == -(-span // 2) and not copied),
                float(n == span and not copied), float(n == span + 2 and not copied),
                float(copied is not None), 0.0, float(c["score"] is not None and c["score"] < REJECTED),
                edit, edit / max(len(source_words), 1), float(np.mean(confidence)), float(np.min(confidence)),
                len(added), len(added & evidence_names), len(added - evidence_names), len(source_names - text_names),
                scorer.score(source, text)["rougeL"].fmeasure, d, len(decisions), len(candidates),
                combined[k], combined[k] - combined.max(),
                sum(w in evidence_words for w in words(copied or "")),
            ])
            texts.append(text)
    # The source row takes the worst logged scores of the pool, so that a tree
    # does not read a zero score as a good one.
    worst = [min(r[i] for r in rows) if rows else 0.0 for i in range(len(FEATURES))]
    row = [0.0] * len(FEATURES)
    for name in ("pll", "ent", "pll_gap", "ent_gap", "combined", "combined_gap"):
        row[FEATURES.index(name)] = worst[FEATURES.index(name)]
    for name in ("source", "source_rougeL", "confidence_mean", "confidence_min"):
        row[FEATURES.index(name)] = 1.0
    row[FEATURES.index("decisions")] = len(decisions)
    row[FEATURES.index("candidates")] = len(rows)
    rows.append(row)
    texts.append(source)
    targets = np.array([scorer.score(target, t)["rougeL"].fmeasure for t in texts])
    return np.array(rows, dtype=float), texts, targets


def pools(run, data, ids=None):
    """One pool per gold EDIT sentence of the articles the run read, as infilling.evaluate scores them."""
    results = list(read_jsonl(Path(run) / "results.jsonl"))
    if ids:
        keep = set(json.loads(Path(ids).read_text()))
        results = [r for r in results if r["id"] in keep]
    wanted = {r["id"] for r in results}
    records = {r["id"]: r for r in read_jsonl(data) if r["id"] in wanted}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    out = []
    for result in results:
        record = records[result["id"]]
        merged = {j: {"decisions": [], "spans": []}
                  for j, entry in enumerate(record["labels"]) if entry.get("label") == "EDIT"}
        for edit in result["edits"]:
            entry = merged.setdefault(edit["sentence"], {"decisions": [], "spans": []})
            entry["decisions"] += edit["decisions"]
            entry["spans"] += edit.get("spans", [])
        for j, edit in merged.items():
            rows, texts, targets = pool(result, record, j, edit["decisions"], edit["spans"], scorer)
            out.append({"id": result["id"], "sentence": j, "rows": rows, "texts": texts, "targets": targets})
    return out


def model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=0)


def fit(args):
    groups = pools(args.run, args.data, args.ids)
    rows = np.vstack([g["rows"] for g in groups])
    best = np.concatenate([(g["targets"] >= g["targets"].max() - 1e-9).astype(float) for g in groups])
    clf = model().fit(rows, best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"model": clf, "features": FEATURES, "run": args.run}, f)
    print(f"{len(groups)} pools, {len(rows)} candidates, {int(best.sum())} best, saved to {args.out}")


def choose(clf, group, no_deletion, no_source):
    scores = clf.predict_proba(group["rows"])[:, 1]
    if no_deletion:
        scores[group["rows"][:, FEATURES.index("zero")] > 0] = -1.0
    if no_source and len(scores) > 1:
        scores[group["rows"][:, FEATURES.index("source")] > 0] = -1.0
    return int(np.argmax(scores))


def apply(args):
    with open(args.model, "rb") as f:
        clf = pickle.load(f)["model"]
    groups = pools(args.run, args.data)
    chosen = {(g["id"], g["sentence"]): g["texts"][choose(clf, g, args.no_deletion, args.no_source)]
              for g in groups}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    changed = 0
    with open(out / "results.jsonl", "w") as f:
        for result in read_jsonl(Path(args.run) / "results.jsonl"):
            for (i, j), text in chosen.items():
                if i == result["id"]:
                    changed += result["output"][j] != text
                    result["output"][j] = text
            f.write(json.dumps(result) + "\n")
    config = json.loads((Path(args.run) / "config.json").read_text())
    config["rerank"] = {"model": args.model, "no_deletion": args.no_deletion, "no_source": args.no_source}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{len(groups)} sentences, {changed} outputs changed, written to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit", help="fit the classifier on the pools of a run")
    p.add_argument("run")
    p.add_argument("data", help="the split the run read, with gold rewrites")
    p.add_argument("out", help="path of the pickled classifier")
    p.add_argument("--ids", help="JSON list of article ids to keep")
    p.set_defaults(func=fit)
    p = sub.add_parser("apply", help="rerank the pools of a run and write a new run directory")
    p.add_argument("model", help="pickled classifier from fit")
    p.add_argument("run")
    p.add_argument("data", help="the split the run read")
    p.add_argument("out", help="the new run directory, scored by infilling.evaluate")
    p.add_argument("--no-deletion", action="store_true", help="never choose a candidate that deletes a span")
    p.add_argument("--no-source", action="store_true", help="never choose the unchanged source sentence")
    p.set_defaults(func=apply)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
