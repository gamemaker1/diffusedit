"""Figures and numbers of the evaluation deep dive, computed from the pulled runs.

Every sentence number is recomputed from results.jsonl and the data split with
the scoring of infilling.evaluate, so that one script produces every figure
and every table entry of the document. Intervals are 95% bootstrap intervals
over articles, with 2,000 resamples.

    ladder.pdf     copy, chosen and best-of-pool ROUGE-L of each single-round run
    gains.pdf      ROUGE-L gain over copy of each run and stress set, with intervals
    rules.pdf      each repair rule removed, ROUGE-L difference from the full setting
    values.pdf     exact match, new-value recall and old value kept per system
    picks.pdf      the kind of candidate the heuristic and the learned ranking choose
    stress.pdf     gain over copy on the clean articles and the three stress sets
    heads.pdf      the heads with and without evidence
    loops.pdf      cascade ROUGE-L and cascade recall of each detector
    numbers.json   every plotted value and the numbers quoted in the text
    examples.json  the candidate pools of the worked examples

The missing-value and FRUIT stress sets are rebuilt under <full>/stress from the
val split and FRUIT train, with the seeds of experiments.stress.

Usage:
    python -m analysis.evaluation --full outputs/jl/full --data ../datasets/synthetic/split \\
        --out ../../docs/internal/evaluation/figures
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np
from rouge_score import rouge_scorer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.values import score as value_rows  # noqa: E402
from classifier.inputs import read_jsonl  # noqa: E402
from experiments import stress  # noqa: E402
from infilling import rerank  # noqa: E402
from infilling.evaluate import gold  # noqa: E402

warnings.filterwarnings("ignore")

INK, MUTED, GRID = "#1C2B33", "#5B6770", "#E3E8EC"
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#9aa3ab"
plt.rcParams.update({
    "font.family": "cmss10", "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": False, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
RESAMPLES = 2000


def sentences(run, data, ids=None):
    """Per gold EDIT sentence of the run: article id, copy, chosen and pool ROUGE-L, and whether it changed."""
    results = list(read_jsonl(Path(run) / "results.jsonl"))
    if ids:
        keep = set(json.loads(Path(ids).read_text()))
        results = [r for r in results if r["id"] in keep]
    wanted = {r["id"] for r in results}
    records = {r["id"]: r for r in read_jsonl(data) if r["id"] in wanted}
    rows = []
    for result in results:
        record = records[result["id"]]
        merged = {j: [] for j, e in enumerate(record["labels"]) if e.get("label") == "EDIT"}
        for edit in result.get("edits", []):
            merged.setdefault(edit["sentence"], []).extend(edit["decisions"])
        # The editor writes whole articles and logs no source and no candidates.
        sources = result.get("source") or record["source_sentences"]
        for j, decisions in merged.items():
            target, source, output = gold(record, j), sources[j], result["output"][j]
            texts = [c["text"] for d in decisions for c in d["candidates"]] or [source]
            rows.append({
                "id": result["id"], "copy": SCORER.score(target, source)["rougeL"].fmeasure,
                "chosen": SCORER.score(target, output)["rougeL"].fmeasure,
                "pool": max(SCORER.score(target, t)["rougeL"].fmeasure for t in texts),
                "changed": output != source,
            })
    return rows


def by_article(rows, key):
    totals, counts = {}, {}
    for r in rows:
        totals[r["id"]] = totals.get(r["id"], 0.0) + r[key]
        counts[r["id"]] = counts.get(r["id"], 0) + 1
    ids = sorted(totals)
    return ids, np.array([totals[i] for i in ids]), np.array([counts[i] for i in ids])


def paired(rows_a, key_a, rows_b, key_b, seed=0):
    """Ratio-of-sums difference A minus B over the shared articles, with its bootstrap interval."""
    ids, a, n = by_article(rows_a, key_a)
    ids_b, b, n_b = by_article(rows_b, key_b)
    shared = [i for i in ids if i in set(ids_b)]
    pa, pb = {i: k for k, i in enumerate(ids)}, {i: k for k, i in enumerate(ids_b)}
    a, n = a[[pa[i] for i in shared]], n[[pa[i] for i in shared]]
    b, n_b = b[[pb[i] for i in shared]], n_b[[pb[i] for i in shared]]
    draws = np.random.default_rng(seed).integers(0, len(shared), (RESAMPLES, len(shared)))
    diffs = a[draws].sum(1) / n[draws].sum(1) - b[draws].sum(1) / n_b[draws].sum(1)
    return a.sum() / n.sum() - b.sum() / n_b.sum(), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def interval_plot(ax, labels, values, lows, highs, colors, xlabel):
    y = np.arange(len(labels))[::-1]
    pad = 0.03 * (max(highs) - min(lows))
    for yi, v, lo, hi, c in zip(y, values, lows, highs, colors):
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.4, solid_capstyle="butt")
        ax.plot(v, yi, "o", color=c, ms=4.5)
        # A label sits on the side of the interval away from zero, so it never crosses the zero line.
        if v >= 0:
            ax.text(hi + pad, yi, f"{v:+.3f}", va="center", fontsize=6.8, color=INK)
        else:
            ax.text(lo - pad, yi, f"{v:+.3f}", va="center", ha="right", fontsize=6.8, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(min(lows + [0]) - 8 * pad, max(highs + [0]) + 8 * pad)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def ladder(runs, out, numbers):
    """Copy, chosen and pool of each single-round run, as a line from copy to pool with a mark at chosen."""
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(runs))[::-1]
    for yi, (label, rows) in zip(y, runs):
        c, ch, p = mean(rows, "copy"), mean(rows, "chosen"), mean(rows, "pool")
        ax.plot([c, p], [yi, yi], color=GRID, lw=6, solid_capstyle="butt", zorder=1)
        ax.plot(c, yi, "|", color=MUTED, ms=11, mew=1.5, zorder=2)
        ax.plot(p, yi, "|", color=MUTED, ms=11, mew=1.5, zorder=2)
        ax.plot(ch, yi, "o", color=BLUE, ms=6, zorder=3)
        ax.text(p + 0.004, yi, f"pool {p:.3f}", va="center", fontsize=6.8, color=MUTED)
        ax.text(ch, yi + 0.34, f"{ch:.3f}", ha="center", fontsize=6.8, color=INK)
        numbers["ladder"][label] = {"copy": c, "chosen": ch, "pool": p, "sentences": len(rows)}
    ax.set_yticks(y, [label for label, _ in runs])
    ax.set_xlim(0.84, 1.0)
    ax.set_xlabel("sentence ROUGE-L against the gold update")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=BLUE, label="chosen candidate")
    ax.plot([], [], "|", color=MUTED, mew=1.5, ms=9, label="copy (left) and best of pool (right)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    fig.savefig(out / "ladder.pdf")
    plt.close(fig)


def gains(items, out, numbers, name, xlabel="ROUGE-L gain over copy"):
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(items) + 0.9))
    labels, values, lows, highs, colors = [], [], [], [], []
    for label, rows, color in items:
        d, lo, hi = paired(rows, "chosen", rows, "copy")
        labels.append(label), values.append(d), lows.append(lo), highs.append(hi), colors.append(color)
        numbers[name][label] = {"gain": d, "low": lo, "high": hi, "sentences": len(rows)}
    interval_plot(ax, labels, values, lows, highs, colors, xlabel)
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def differences(items, out, numbers, name, xlabel):
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(items) + 0.9))
    labels, values, lows, highs, colors = [], [], [], [], []
    for label, rows, reference, color in items:
        d, lo, hi = paired(rows, "chosen", reference, "chosen")
        labels.append(label), values.append(d), lows.append(lo), highs.append(hi), colors.append(color)
        numbers[name][label] = {"difference": d, "low": lo, "high": hi, "sentences": len(rows)}
    interval_plot(ax, labels, values, lows, highs, colors, xlabel)
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def values(systems, records, out, numbers):
    """Exact match, new-value recall and old value kept per system, one panel per measure."""
    keys = ("exact", "new", "old kept")
    titles = ("(a) Exact match", "(b) New-value recall", "(c) Old value kept")
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.2), sharey=True)
    y = np.arange(len(systems))[::-1]
    for ax, k, t in zip(axes, range(3), titles):
        for yi, (label, outputs, color) in zip(y, systems):
            per = [value_rows(records[i], outputs[i]) for i in sorted(records) if i in outputs]
            s = np.array([np.nansum([r[k] for r in rows]) for rows in per])
            c = np.array([np.sum([not np.isnan(r[k]) for r in rows]) for rows in per])
            draws = np.random.default_rng(0).integers(0, len(s), (RESAMPLES, len(s)))
            means = s[draws].sum(1) / c[draws].sum(1)
            v, lo, hi = s.sum() / c.sum(), np.percentile(means, 2.5), np.percentile(means, 97.5)
            ax.barh(yi, v, color=color, height=0.62, alpha=0.85)
            ax.plot([lo, hi], [yi, yi], color=INK, lw=0.8)
            ax.text(max(v, hi) + 0.02, yi, f"{v:.2f}", va="center", fontsize=6.8)
            numbers["values"].setdefault(label, {})[keys[k]] = {"value": float(v), "low": float(lo), "high": float(hi)}
        ax.set_title(t, loc="left")
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.5, 1.0])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [label for label, _, _ in systems])
    fig.tight_layout()
    fig.savefig(out / "values.pdf")
    plt.close(fig)


def kinds(groups, picks, numbers, name):
    """Count the kind of candidate each pick is."""
    counts = {"evidence copy": 0, "sampled, same length": 0, "sampled, shorter": 0, "sampled, longer": 0,
              "deletion": 0, "unchanged": 0}
    zero, half, same, plus2, copied, source = (rerank.FEATURES.index(f) for f in
                                               ("zero", "half", "same", "plus2", "copied", "source"))
    for g, p in zip(groups, picks):
        row = g["rows"][p]
        if row[source] or g["texts"][p] == g["texts"][-1]:
            counts["unchanged"] += 1
        elif row[copied]:
            counts["evidence copy"] += 1
        elif row[zero]:
            counts["deletion"] += 1
        elif row[same]:
            counts["sampled, same length"] += 1
        elif row[half]:
            counts["sampled, shorter"] += 1
        elif row[plus2]:
            counts["sampled, longer"] += 1
        else:
            counts["sampled, same length"] += 1
    numbers["picks"][name] = counts
    return counts


def picks(full, data, out, numbers):
    groups = rerank.pools(full / "runs/fix-tuned", data / "test.jsonl")
    outputs = {}
    for result in read_jsonl(full / "runs/fix-rerank/results.jsonl"):
        for j, t in enumerate(result["output"]):
            outputs[(result["id"], j)] = t
    logged = {}
    for result in read_jsonl(full / "runs/fix-tuned/results.jsonl"):
        for j, t in enumerate(result["output"]):
            logged[(result["id"], j)] = t
    heuristic = [g["texts"].index(logged[(g["id"], g["sentence"])]) for g in groups]
    learned = [g["texts"].index(outputs[(g["id"], g["sentence"])]) for g in groups]
    before = kinds(groups, heuristic, numbers, "heuristic")
    after = kinds(groups, learned, numbers, "learned")
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    colors = [AQUA, BLUE, "#7fb3e6", "#1f5aa6", ORANGE, GRAY]
    for yi, (label, counts) in enumerate((("learned ranking", after), ("heuristic ranking", before))):
        left = 0
        for (kind, n), c in zip(counts.items(), colors):
            ax.barh(yi, n, left=left, color=c, height=0.6, edgecolor="white", lw=0.8)
            if n >= 90:
                ax.text(left + n / 2, yi, str(n), ha="center", va="center", fontsize=6.8, color="white")
            left += n
    ax.set_yticks([0, 1], ["learned ranking", "heuristic ranking"])
    ax.set_xlabel("EDIT sentences of the supported test articles")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, list(before), loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3)
    fig.tight_layout()
    fig.savefig(out / "picks.pdf")
    plt.close(fig)
    numbers["picks"]["sentences"] = len(groups)


def reranked_stress(full, data, numbers):
    """Fit on the clean val pools outside a fold, apply to the stress pools inside it."""
    clean = rerank.pools(full / "runs/repair-val2-v2", data / "val.jsonl")
    ids = sorted({g["id"] for g in clean})
    stress_dir = full / "stress"
    stress_dir.mkdir(exist_ok=True)
    (stress_dir / "ids.json").write_text(json.dumps(ids))
    stress.missing(argparse.Namespace(data=str(data / "val.jsonl"), ids=str(stress_dir / "ids.json"), out=str(stress_dir)))
    sets = {"missing": rerank.pools(full / "runs/stress-missing-chosen", stress_dir / "missing.jsonl"),
            "oldvalue": rerank.pools(full / "runs/stress-oldvalue-chosen", data / "val.jsonl"),
            "clean": clean}
    rows = np.vstack([g["rows"] for g in clean])
    best = np.concatenate([(g["targets"] >= g["targets"].max() - 1e-9).astype(float) for g in clean])
    article = np.concatenate([[g["id"]] * len(g["targets"]) for g in clean])
    result = {}
    for name, groups in sets.items():
        gains, seeds = [], (0, 1, 2)
        for seed in seeds:
            fold = {a: k % 5 for k, a in enumerate(np.random.default_rng(seed).permutation(ids))}
            chosen = []
            models = {}
            for g in groups:
                k = fold[g["id"]]
                if k not in models:
                    train = np.array([fold[a] != k for a in article])
                    models[k] = rerank.model().fit(rows[train], best[train])
                chosen.append(rerank.choose(models[k], g, False, False))
            gains.append(np.mean([g["targets"][p] - g["targets"][-1] for g, p in zip(groups, chosen)]))
        result[name] = float(np.mean(gains))
    numbers["reranked_stress"] = result
    return result


def heads(full, out, numbers):
    n = json.loads((full / "analysis/numbers.json").read_text())
    items = [("Sentence classifier, macro-F1", [("block 24", n["classifier"]["ce24"]["macro_f1"], BLUE),
                                                ("block 24, no evidence", n["classifier"]["noev24"]["macro_f1"], ORANGE),
                                                ("MiniLM baseline", n["classifier"]["baseline"]["macro_f1"], GRAY)]),
             ("Staleness head, PR-AUC", [("attention head", n["staleness"]["attention"]["pr_auc"], BLUE),
                                         ("attention, evidence of another article", n["staleness"]["controls"]["shuffled_evidence"]["pr_auc"], ORANGE),
                                         ("attention, no evidence", n["staleness"]["controls"]["no_evidence"]["pr_auc"], ORANGE),
                                         ("linear head", n["staleness"]["linear"]["pr_auc"], BLUE),
                                         ("linear, no evidence", n["staleness"]["noev"]["pr_auc"], ORANGE)])]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.1), gridspec_kw={"width_ratios": [1, 1.3]})
    for ax, (title, rows) in zip(axes, items):
        y = np.arange(len(rows))[::-1]
        for yi, (label, v, c) in zip(y, rows):
            ax.barh(yi, v, color=c, height=0.6, alpha=0.85)
            ax.text(v + 0.02, yi, f"{v:.3f}", va="center", fontsize=6.8)
            numbers["heads"][label] = v
        ax.set_yticks(y, [label for label, _, _ in rows])
        ax.set_xlim(0, 1.12)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "heads.pdf")
    plt.close(fig)


def loops(full, out, numbers):
    m = json.loads((full / "analysis/loop_metrics.json").read_text())["means"]
    detectors = [("none", "No detector, one round"), ("staleness_delta", "Staleness delta"),
                 ("edited_as_evidence", "Edited as evidence"), ("consequence", "Consequence head"),
                 ("oracle-labels", "Consequence, gold labels")]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.0), sharey=True)
    y = np.arange(len(detectors))[::-1]
    panels = (("(a) Cascade ROUGE-L", "cascade_rougeL"), ("(b) UpdateROUGE-L", "rougeL"), ("(c) KEEP verbatim", "keep_verbatim"))
    for ax, (title, key) in zip(axes, panels):
        for yi, (k, label) in zip(y, detectors):
            v = m[k][key]
            ax.barh(yi, v, color=BLUE if k == "consequence" else GRAY, height=0.6, alpha=0.85)
            ax.text(v + 0.02, yi, f"{v:.3f}", va="center", fontsize=6.8)
            numbers["loops"].setdefault(label, {})[key] = v
        ax.set_title(title, loc="left")
        ax.set_xlim(0, 1.2)
        ax.set_xticks([0, 0.5, 1.0])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [label for _, label in detectors])
    fig.tight_layout()
    fig.savefig(out / "loops.pdf")
    plt.close(fig)
    n = json.loads((full / "analysis/numbers.json").read_text())["cascades"]
    numbers["cascade_recall"] = {k: n[k]["recall"][0] for k in ("mention", "staleness_delta", "edited_as_evidence", "consequence")}


def examples(full, data, out, numbers):
    """The candidate pools of the worked examples, with the heuristic and the learned choice."""
    wanted = {("synthetic-000294", 0), ("synthetic-000722", 1)}
    groups = [g for g in rerank.pools(full / "runs/fix-tuned", data / "test.jsonl") if (g["id"], g["sentence"]) in wanted]
    logged = {}
    for run in ("fix-tuned", "fix-rerank"):
        for result in read_jsonl(full / f"runs/{run}/results.jsonl"):
            for j, t in enumerate(result["output"]):
                logged[(run, result["id"], j)] = t
    records = {r["id"]: r for r in read_jsonl(data / "test.jsonl") if r["id"] in {i for i, _ in wanted}}
    raw = {}
    for result in read_jsonl(full / "runs/fix-tuned/results.jsonl"):
        if result["id"] in records:
            raw[result["id"]] = result
    dump = []
    for g in groups:
        record = records[g["id"]]
        edits = [e for e in raw[g["id"]]["edits"] if e["sentence"] == g["sentence"]]
        dump.append({
            "id": g["id"], "sentence": g["sentence"], "source": g["texts"][-1], "target": gold(record, g["sentence"]),
            "evidence": record["evidence"], "masked": [m for e in edits for m in e["masked"]],
            "heuristic": logged[("fix-tuned", g["id"], g["sentence"])], "learned": logged[("fix-rerank", g["id"], g["sentence"])],
            "candidates": [{"text": c["text"], "lengths": c["lengths"], "pll": c["pll"], "ent": c["ent"], "score": c["score"],
                            "copied": c.get("copied"), "deletion": c["deletion"]}
                           for e in edits for d in e["decisions"] for c in d["candidates"]],
            "rougeL": [float(t) for t in g["targets"]],
        })
    (out / "examples.json").write_text(json.dumps(dump, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", default="outputs/jl/full")
    parser.add_argument("--data", default="../datasets/synthetic/split")
    parser.add_argument("--out", default="../../docs/internal/evaluation/figures")
    args = parser.parse_args()
    full, data, out = Path(args.full), Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = full / "runs"
    supported = full / "supported_ids.json"
    numbers = {k: {} for k in ("ladder", "gains", "rules", "lengths", "values", "picks", "heads", "loops", "stress", "reranker")}

    test = {name: sentences(runs / name, data / "test.jsonl", supported) for name in
            ("fix-oracle", "fix-oracle-noev", "fix-predicted", "fix-attention", "fix-tuned", "fix-rerank")}
    sample = {name: sentences(runs / name, data / "test.jsonl") for name in
              ("fix-oracle-tuned", "fix-oracle-rerank", "abl-no-penalty", "abl-no-reject", "abl-no-veto",
               "abl-no-cap", "abl-no-copy", "abl-same", "abl-pll", "abl-verifier")}
    sample_ids = {r["id"] for r in sample["fix-oracle-tuned"]}
    restrict = lambda rows: [r for r in rows if r["id"] in sample_ids]  # noqa: E731
    val = {name: sentences(runs / name, data / "val.jsonl") for name in ("repair-val2-base", "repair-val2-v1", "repair-val2-v2")}
    stress_runs = {name: sentences(runs / name, data / "val.jsonl") for name in
                   ("stress-missing-base", "stress-missing-chosen", "stress-oldvalue-base", "stress-oldvalue-chosen")}
    stress_dir = full / "stress"
    stress_dir.mkdir(exist_ok=True)
    stress.fruit(argparse.Namespace(data=str(data / "../../collated/train.jsonl"), out=str(stress_dir)))
    fruit_ids = set(json.loads((stress_dir / "fruit_ids.json").read_text()))
    run_ids = {r["id"] for r in read_jsonl(runs / "stress-fruit-chosen/results.jsonl")}
    fruit = {name: sentences(runs / name, stress_dir / "fruit.jsonl") if fruit_ids == run_ids else None
             for name in ("stress-fruit-base", "stress-fruit-chosen")}
    editor = sentences(runs / "editor", data / "test.jsonl", supported)

    ladder([("Oracle masks, base", test["fix-oracle"]), ("Oracle masks, no evidence", test["fix-oracle-noev"]),
            ("Linear-head masks, base", test["fix-predicted"]), ("Attention-head masks, base", test["fix-attention"]),
            ("Attention-head masks, v2", test["fix-tuned"]), ("Attention-head masks, v2, learned ranking", test["fix-rerank"]),
            ("Oracle masks, v2 (sample)", sample["fix-oracle-tuned"]),
            ("Oracle masks, v2, learned ranking (sample)", sample["fix-oracle-rerank"])], out, numbers)
    gains([("Oracle masks, base", test["fix-oracle"], BLUE), ("Oracle masks, no evidence", test["fix-oracle-noev"], ORANGE),
           ("Linear-head masks, base", test["fix-predicted"], BLUE), ("Attention-head masks, base", test["fix-attention"], BLUE),
           ("Attention-head masks, v2", test["fix-tuned"], BLUE), ("Attention-head masks, v2, learned ranking", test["fix-rerank"], BLUE),
           ("Oracle masks, v2 (sample)", sample["fix-oracle-tuned"], VIOLET),
           ("Oracle masks, v2, learned ranking (sample)", sample["fix-oracle-rerank"], VIOLET),
           ("T5 editor", editor, AQUA)], out, numbers, "gains")
    reference = restrict(test["fix-tuned"])
    differences([("No edit penalty", sample["abl-no-penalty"], reference, BLUE),
                 ("No rejection of unsupported words", sample["abl-no-reject"], reference, BLUE),
                 ("No evidence veto", sample["abl-no-veto"], reference, BLUE),
                 ("No span cap", sample["abl-no-cap"], reference, BLUE),
                 ("No evidence-copy candidates", sample["abl-no-copy"], reference, BLUE),
                 ("Learned ranking", restrict(test["fix-rerank"]), reference, AQUA)],
                out, numbers, "rules", "sentence ROUGE-L minus the v2 setting (557 sentences)")
    oracle = restrict(test["fix-oracle"])
    differences([("Same-length candidates only", sample["abl-same"], oracle, BLUE),
                 ("PLL-only ranking", sample["abl-pll"], oracle, BLUE),
                 ("Verifier-only ranking", sample["abl-verifier"], oracle, BLUE)],
                out, numbers, "lengths", "sentence ROUGE-L minus the base setting with oracle masks (557 sentences)")

    records = {r["id"]: r for r in read_jsonl(data / "test.jsonl") if r["id"] in set(json.loads(supported.read_text()))}
    outputs = {}
    for name in ("fix-tuned", "fix-rerank", "editor"):
        outputs[name] = {r["id"]: r["output"] for r in read_jsonl(runs / name / "results.jsonl") if r["id"] in records}
    outputs["copy"] = {i: r["source_sentences"] for i, r in records.items()}
    values([("Copy", outputs["copy"], GRAY), ("Heuristic ranking (fix-tuned)", outputs["fix-tuned"], BLUE),
            ("Learned ranking (fix-rerank)", outputs["fix-rerank"], AQUA), ("T5 editor", outputs["editor"], VIOLET)],
           records, out, numbers)
    picks(full, data, out, numbers)

    cv = reranked_stress(full, data, numbers)
    items = [("Clean val articles, base", val["repair-val2-base"], GRAY), ("Clean val articles, v2", val["repair-val2-v2"], BLUE),
             ("Missing value, base", stress_runs["stress-missing-base"], GRAY), ("Missing value, v2", stress_runs["stress-missing-chosen"], BLUE),
             ("Old value in a passage, base", stress_runs["stress-oldvalue-base"], GRAY),
             ("Old value in a passage, v2", stress_runs["stress-oldvalue-chosen"], BLUE)]
    if fruit["stress-fruit-base"]:
        items += [("FRUIT train, base", fruit["stress-fruit-base"], GRAY), ("FRUIT train, v2", fruit["stress-fruit-chosen"], BLUE)]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(items) + 1.3))
    labels, vals, lows, highs, colors = [], [], [], [], []
    for label, rows, color in items:
        d, lo, hi = paired(rows, "chosen", rows, "copy")
        labels.append(label), vals.append(d), lows.append(lo), highs.append(hi), colors.append(color)
        numbers["stress"][label] = {"gain": d, "low": lo, "high": hi, "sentences": len(rows)}
    interval_plot(ax, labels, vals, lows, highs, colors, "ROUGE-L gain over copy")
    y = np.arange(len(labels))[::-1]
    for label, key in (("Clean val articles, v2", "clean"), ("Missing value, v2", "missing"), ("Old value in a passage, v2", "oldvalue")):
        yi = y[labels.index(label)]
        ax.plot(cv[key], yi - 0.3, "D", color=AQUA, ms=4)
        ax.text(cv[key] + 0.003, yi - 0.3, f"learned ranking {cv[key]:+.3f}", va="center", fontsize=6.4, color=AQUA)
    fig.tight_layout()
    fig.savefig(out / "stress.pdf")
    plt.close(fig)

    heads(full, out, numbers)
    loops(full, out, numbers)
    examples(full, data, out, numbers)

    numbers["reranker"]["test"] = dict(zip(("difference", "low", "high"), paired(test["fix-rerank"], "chosen", test["fix-tuned"], "chosen")))
    numbers["reranker"]["oracle_sample"] = dict(zip(("difference", "low", "high"),
                                                    paired(sample["fix-oracle-rerank"], "chosen", sample["fix-oracle-tuned"], "chosen")))
    numbers["reranker"]["editor_minus_rerank"] = dict(zip(("difference", "low", "high"), paired(editor, "chosen", test["fix-rerank"], "chosen")))
    numbers["reranker"]["editor_minus_tuned"] = dict(zip(("difference", "low", "high"), paired(editor, "chosen", test["fix-tuned"], "chosen")))
    numbers["reranker"]["changed"] = {name: mean(rows, "changed") for name, rows in (("fix-tuned", test["fix-tuned"]), ("fix-rerank", test["fix-rerank"]))}
    for name, rows in list(test.items()) + list(sample.items()) + list(val.items()) + list(stress_runs.items()):
        numbers.setdefault("runs", {})[name] = {"copy": mean(rows, "copy"), "chosen": mean(rows, "chosen"),
                                                "pool": mean(rows, "pool"), "changed": mean(rows, "changed"), "sentences": len(rows)}
    numbers["runs"]["editor"] = {"copy": mean(editor, "copy"), "chosen": mean(editor, "chosen"), "sentences": len(editor)}
    (out / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps({k: numbers[k] for k in ("reranker", "reranked_stress")}, indent=1))


if __name__ == "__main__":
    main()
