"""Figures and numbers of the full synthetic run, scored on the test split.

Reads the outputs of jobs/jl/run_full.sh and jobs/jl/report.sh under one
directory (heads, noev/heads, runs and analysis). Nothing here uses the GPU,
and a figure whose inputs do not exist yet is skipped. The style is that of
analysis.progress_figures.

    classifier.pdf   test macro-F1 of every sentence head, and per-class F1
    confusion.pdf    confusion matrices on test of the baseline and the block-24 head
    seeds.pdf        five seeds of each classifier and staleness configuration
    staleness.pdf    precision against recall of the token heads, and the attention controls
    cascades.pdf     recall and precision of the cascade detectors, recall per kind, the tau2 sweep
    infilling.pdf    hit, ROUGE-L and entity measures of the single-round runs
    fix_differences.pdf   paired differences between single-round systems, 95% intervals
    loop.pdf         cascade measures and rounds of the loop runs
    loop_differences.pdf  paired differences between loop runs, 95% intervals
    ablations.pdf    every ablation as the full component minus the ablated one, one panel
                     per contribution, with 95% intervals where the outputs allow a test.
                     The pooling rows read analysis/classifier_tests.json and the token-head
                     rows analysis/stale_tests.json, and fall back to the point difference
                     or the seed interval while those tests have not run.

The numbers go to <full>/analysis/numbers.json.

Usage:
    python -m analysis.results --full /home/diffusedit/full
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter

from analysis.classifier_tests import decide, macro_f1
from analysis.progress_figures import GRID, INK, LABELS, MUTED, SEEDS, bar, fill, load, plt, stroke, title, wilson

plt.rcParams["axes.formatter.use_mathtext"] = False

CLASSIFIERS = (("baseline", "MiniLM\nbaseline", "gray"), ("noev24", "Block 24,\nno evidence", "red"),
               ("ce32", "Block 32", "blue"), ("ce24", "Block 24", "blue"),
               ("mean", "+ evidence\nmean", "green"), ("nearest", "+ evidence\nnearest", "green"),
               ("attention", "+ evidence\nattention", "green"))
FIXES = (("oracle", "Oracle\nmasks"), ("oracle-noev", "Oracle,\nno evidence"), ("predicted", "Linear-head\nmasks"),
         ("attention", "Attention-head\nmasks"), ("tuned", "Attention,\nval-tuned"))
DETECTORS = (("mention", "Shared\nmention"), ("staleness_delta", "Staleness\ndelta"),
             ("edited_as_evidence", "Edited as\nevidence"), ("consequence", "Consequence\nhead"))
LOOPS = (("none", "No detector"), ("mention", "Shared mention"), ("staleness_delta", "Staleness delta"),
         ("edited_as_evidence", "Edited as evidence"), ("consequence", "Consequence head"),
         ("oracle-labels", "Consequence, gold labels"), ("noev", "Consequence, no evidence"))


def maybe(path):
    path = Path(path)
    return load(path) if path.exists() else None


def predictions(path):
    d = np.load(path)
    return d["labels"].astype(np.int64), decide(d["probs"].astype(np.float32), float(d["tau_d"]))


def classifier_paths(full):
    sweep = full / "heads/sweep"
    return {"baseline": full / "heads/baseline", "ce24": sweep / "cls-ce/layer24", "ce32": sweep / "cls-ce/layer32",
            "focal24": sweep / "cls-focal/layer24", "focal32": sweep / "cls-focal/layer32",
            "noev24": full / "noev/heads/sweep/cls-ce/layer24"}


def seed_scores(full):
    out = {}
    for name, path in classifier_paths(full).items():
        files = [path / f"seed{s}/test_predictions.npz" for s in SEEDS]
        if all(f.exists() for f in files):
            out[name] = [macro_f1(*predictions(f)) for f in files]
    for name, path in (("stale-bce", "heads/sweep/stale-bce"), ("stale-focal", "heads/sweep/stale-focal"),
                       ("noev-stale-bce", "noev/heads/sweep/stale-bce")):
        files = [full / path / f"seed{s}/metrics.json" for s in SEEDS]
        if all(f.exists() for f in files):
            out[name] = [load(f)["test"]["pr_auc"] for f in files]
    return out


def classifier(full, seeds, out):
    single = {"baseline": "heads/baseline", "noev24": "noev/heads/classifier/layer24",
              "ce32": "heads/classifier/layer32", "ce24": "heads/classifier/layer24",
              "mean": "heads/evidence_mean/layer24", "nearest": "heads/evidence_nearest/layer24",
              "attention": "heads/evidence_attention/layer24"}
    tests = {k: maybe(full / p / "metrics.json") for k, p in single.items()}
    tests = {k: v["test"] for k, v in tests.items() if v}
    rows = [c for c in CLASSIFIERS if c[0] in tests]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
    for k, (key, _, color) in enumerate(rows):
        v = tests[key]["macro_f1"]
        bar(ax1, k, v, color, width=0.62)
        if key in seeds:
            s = np.array(seeds[key])
            ax1.errorbar(k, s.mean(), yerr=s.std(ddof=1), color=INK, capsize=2, lw=0.8)
        ax1.text(k, v + 0.02, f"{v:.3f}", ha="center", fontsize=6.4)
    ax1.axhline(1 / 3, color=MUTED, lw=0.7, ls=":")
    ax1.set_xticks(range(len(rows)), [name for _, name, _ in rows], fontsize=6.4)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("test macro-F1")
    title(ax1, "(a) Sentence heads on test")
    shown = [r for r in rows if r[0] in ("baseline", "noev24", "ce24", "attention")]
    width = 0.8 / max(len(shown), 1)
    for i, (key, name, color) in enumerate(shown):
        f1 = [tests[key][label]["f1"] for label in LABELS]
        bar(ax2, np.arange(3) + (i - (len(shown) - 1) / 2) * width, f1, color, width=width,
            hatch="////" if key == "noev24" else None, label=name.replace("\n", " "))
    ax2.set_xticks(range(3), LABELS)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("test F1")
    ax2.legend(fontsize=6, loc="upper right")
    title(ax2, "(b) F1 per label")
    fig.tight_layout()
    fig.savefig(out / "classifier.pdf")
    plt.close(fig)
    return tests


def confusion(full, out):
    paths = classifier_paths(full)
    shown = [(k, n) for k, n in (("baseline", "MiniLM baseline"), ("noev24", "Block 24, no evidence"),
                                 ("ce24", "Block 24")) if (paths[k] / "seed42/test_predictions.npz").exists()]
    if not shown:
        return None
    fig, axes = plt.subplots(1, len(shown), figsize=(2.3 * len(shown), 2.2))
    axes = np.atleast_1d(axes)
    numbers = {}
    for ax, (key, name) in zip(axes, shown):
        y, pred = predictions(paths[key] / "seed42/test_predictions.npz")
        m = np.zeros((3, 3), int)
        np.add.at(m, (y, pred), 1)
        share = m / m.sum(1, keepdims=True)
        numbers[key] = m.tolist()
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{share[i, j]:.2f}\n{m[i, j]:,}", ha="center", va="center", fontsize=6,
                        color="white" if share[i, j] > 0.6 else INK)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("predicted")
        ax.grid(False)
        title(ax, name)
    axes[0].set_ylabel("gold")
    fig.tight_layout()
    fig.savefig(out / "confusion.pdf")
    plt.close(fig)
    return numbers


def seeds_figure(seeds, out):
    if not seeds:
        return
    groups = [(("baseline", "MiniLM"), ("noev24", "24, no ev."), ("ce32", "32, CE"), ("focal32", "32, focal"),
               ("ce24", "24, CE"), ("focal24", "24, focal")),
              (("noev-stale-bce", "BCE, no ev."), ("stale-focal", "Focal"), ("stale-bce", "BCE"))]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), gridspec_kw={"width_ratios": [2, 1]})
    for ax, group, label, name in zip(axes, groups, ("test macro-F1", "test PR-AUC"),
                                      ("(a) Classifier, five seeds", "(b) Linear staleness head, five seeds")):
        group = [g for g in group if g[0] in seeds]
        for k, (key, _) in enumerate(group):
            v = np.array(seeds[key])
            ax.scatter(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, s=12, color=fill("blue"),
                       edgecolor=stroke("blue"), linewidth=0.6, zorder=3)
            ax.text(k, v.max() + 0.03, f"{v.mean():.3f}\nsd {v.std(ddof=1):.3f}", ha="center", fontsize=6)
        ax.set_xticks(range(len(group)), [n for _, n in group], fontsize=6.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel(label)
        title(ax, name)
    fig.tight_layout()
    fig.savefig(out / "seeds.pdf")
    plt.close(fig)


def pr_curve(labels, probs):
    order = np.argsort(-probs.astype(np.float32), kind="stable")
    y = labels[order].astype(float)
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / max(y.sum(), 1)
    step = max(len(y) // 2000, 1)
    return recall[::step], precision[::step]


def staleness(full, out):
    files = {"linear": full / "heads/sweep/stale-bce/seed42/test_predictions.npz",
             "linear, no evidence": full / "noev/heads/sweep/stale-bce/seed42/test_predictions.npz"}
    linear = maybe(full / "heads/staleness/metrics.json")
    attention = maybe(full / "heads/staleness_attention/metrics.json")
    if not linear:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.3, 1]})
    numbers = {"linear": linear["test"], "noev": (maybe(full / "noev/heads/staleness/metrics.json") or {}).get("test")}
    for (name, path), color in zip(files.items(), ("blue", "red")):
        if path.exists():
            d = np.load(path)
            r, p = pr_curve(d["labels"], d["probs"])
            ax1.plot(r, p, color=stroke(color), lw=1, label=name)
            numbers.setdefault("base_rate", float(d["labels"].mean()))
    for metrics, color, name in ((linear, "blue", "linear at tau"), (attention, "green", "attention at tau")):
        if metrics:
            ax1.scatter(metrics["test"]["recall"], metrics["test"]["precision"], color=fill(color),
                        edgecolor=stroke(color), zorder=4, s=22, label=name)
    if "base_rate" in numbers:
        ax1.axhline(numbers["base_rate"], color=MUTED, lw=0.7, ls=":")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.legend(fontsize=6, loc="lower left")
    title(ax1, "(a) Stale tokens on test")
    if attention:
        numbers["attention"] = attention["test"]
        numbers["controls"] = attention.get("controls")
        controls = attention.get("controls") or {}
        rows = [("test", "With evidence", attention["test"])]
        rows += [(k, n, controls[k]) for k, n in (("shuffled_evidence", "Shuffled\nevidence"),
                                                   ("no_evidence", "No\nevidence"))
                 if isinstance(controls.get(k), dict)]
        for k, (_, name, m) in enumerate(rows):
            v = m.get("pr_auc")
            if v is None:
                continue
            bar(ax2, k, v, "green" if k == 0 else "red", width=0.6)
            ax2.text(k, v + 0.02, f"{v:.3f}", ha="center", fontsize=6.4)
        ax2.set_xticks(range(len(rows)), [n for _, n, _ in rows], fontsize=6.5)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("test PR-AUC")
        title(ax2, "(b) Attention head, evidence controls")
    fig.tight_layout()
    fig.savefig(out / "staleness.pdf")
    plt.close(fig)
    return numbers


def cascades(full, out):
    runs = {k: maybe(full / f"runs/{k}.json") for k, _ in DETECTORS}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return None
    shown = [(k, n) for k, n in DETECTORS if k in runs]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.1, 1.7, 0.8]})
    numbers = {}
    for k, (key, _) in enumerate(shown):
        m = runs[key]
        n = sum(g["cascades"] for g in (m["group_a_shared_mention"], m["group_b_no_shared_mention"]))
        rec = wilson(m["flagged_cascades"], n)
        prec = wilson(m["flagged_cascades"], m["flagged"]) if m["flagged"] else (0, 0, 0)
        numbers[key] = {"recall": rec, "precision": prec, "cascades": n, "flagged": m["flagged"],
                        "shared": m["group_a_shared_mention"], "unshared": m["group_b_no_shared_mention"],
                        "tau2": m.get("tau2"), "tau_c": m.get("tau_c")}
        for offset, (v, color) in zip((-0.19, 0.19), ((rec, "blue"), (prec, "orange"))):
            bar(ax1, k + offset, v[0], color, width=0.36, label=None if k else ("recall" if color == "blue" else "precision"))
            ax1.errorbar(k + offset, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color=INK, capsize=1.5, lw=0.7)
    ax1.set_xticks(range(len(shown)), [n for _, n in shown], fontsize=6.3)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=6, loc="upper left")
    title(ax1, "(a) Cascade sentences on test")
    kinds = sorted({k for m in runs.values() for k in m.get("by_kind", {}) if k != "dates (pooled)"})
    width = 0.8 / len(shown)
    colors = ("gray", "yellow", "purple", "green")
    for i, (key, name) in enumerate(shown):
        by_kind = runs[key].get("by_kind", {})
        v = [by_kind.get(k, {}).get("recall") or 0 for k in kinds]
        bar(ax2, np.arange(len(kinds)) + (i - (len(shown) - 1) / 2) * width, v, colors[i % 4], width=width,
            label=name.replace("\n", " "))
    counts = {k: next(m["by_kind"][k]["cascades"] for m in runs.values() if k in m.get("by_kind", {})) for k in kinds}
    ax2.set_xticks(range(len(kinds)), [f"{k}\n({counts[k]})" for k in kinds], fontsize=5.8, rotation=30, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("recall")
    ax2.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, columnspacing=0.8,
               handlelength=1.2)
    title(ax2, "(b) Recall per cascade kind")
    numbers["by_kind"] = {key: runs[key].get("by_kind") for key, _ in shown}
    sweep = maybe(full / "runs/tau2/chosen.json")
    if sweep:
        t = sorted(sweep["val"], key=float)
        for metric, color in (("recall", "blue"), ("precision", "orange"), ("f1", "green")):
            ax3.plot(range(len(t)), [sweep["val"][x][metric] for x in t], marker="o", ms=3,
                     color=stroke(color), lw=1, label=metric)
        ax3.axvline(t.index(next(x for x in t if float(x) == float(sweep["tau2"]))), color=MUTED, lw=0.7, ls=":")
        ax3.set_xticks(range(len(t)), [f"{float(x):g}" for x in t], fontsize=6.3)
        ax3.set_xlabel(r"$\tau_2$")
        ax3.set_ylim(0, 1.05)
        ax3.legend(fontsize=6)
        title(ax3, "(c) Staleness delta on val")
        numbers["tau2_sweep"] = sweep
    fig.tight_layout()
    fig.savefig(out / "cascades.pdf")
    plt.close(fig)
    return numbers


def infilling(full, out):
    runs = {k: maybe(full / f"runs/fix-{k}/metrics.json") for k, _ in FIXES}
    runs = {k: v for k, v in runs.items() if v}
    articles = maybe(full / "analysis/fix_metrics.json")
    if not runs:
        return None
    shown = [(k, n) for k, n in FIXES if k in runs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    width = 0.8 / (len(shown) + 1)
    measures = (("hit", "Gold-word\nhit"), ("chosen_rougeL", "ROUGE-L of\nrepaired sentences"))
    for i, (key, name) in enumerate([("copy", "Copy")] + shown):
        if key == "copy":
            v = [0.0, runs[shown[0][0]]["copy_rougeL"]]
        else:
            v = [runs[key]["hit"], runs[key]["chosen_rougeL"]]
        x = np.arange(len(measures)) + (i - len(shown) / 2) * width
        bar(ax1, x, v, ("gray", "blue", "red", "orange", "green", "purple")[i], width=width, label=name.replace("\n", " "))
        for xx, vv in zip(x, v):
            ax1.text(xx, vv + 0.015, f"{vv:.2f}", ha="center", fontsize=5.2)
    ax1.set_xticks(range(len(measures)), [n for _, n in measures])
    ax1.set_ylim(0, 1.08)
    ax1.legend(fontsize=5.8, loc="upper left")
    title(ax1, "(a) Repaired sentences")
    if articles:
        means = articles["means"]
        systems = [("copy", "Copy")] + [(k, n) for k, n in FIXES if k in means]
        metric_names = (("rougeL", "UpdateROUGE-L"), ("entity_precision", "Entity\nprecision"),
                        ("entity_recall", "Entity\nrecall"), ("keep_verbatim", "KEEP\nverbatim"))
        width = 0.8 / len(systems)
        for i, (key, name) in enumerate(systems):
            v = [means[key].get(m) or 0 for m, _ in metric_names]
            bar(ax2, np.arange(len(metric_names)) + (i - (len(systems) - 1) / 2) * width, v,
                ("gray", "blue", "red", "orange", "green", "purple")[i], width=width, label=name.replace("\n", " "))
        ax2.set_xticks(range(len(metric_names)), [n for _, n in metric_names], fontsize=6.5)
        ax2.set_ylim(0, 1.05)
        title(ax2, f"(b) Articles ({articles['articles']})")
    fig.tight_layout()
    fig.savefig(out / "infilling.pdf")
    plt.close(fig)
    return {"runs": runs, "articles": articles and {k: articles[k] for k in ("articles", "means", "counted")}}


def differences(path, out, name, metrics):
    report = maybe(path)
    if not report or not report.get("tests"):
        return None
    rows = [r for r in report["tests"] if r["metric"] in metrics]
    fig, ax = plt.subplots(figsize=(6.8, 0.22 * len(rows) + 0.8))
    for k, r in enumerate(rows):
        significant = r["p_holm"] < 0.05
        color = "blue" if significant else "gray"
        ax.errorbar(r["diff"], k, xerr=[[r["diff"] - r["ci_low"]], [r["ci_high"] - r["diff"]]], fmt="o", ms=3,
                    color=stroke(color), capsize=1.5, lw=0.8)
        ax.text(1.01, k, f"{r['diff']:+.3f}  Holm p {r['p_holm']:.3g}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=5.8, color=INK if significant else MUTED)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks(range(len(rows)), [f"{r['pair']}  {r['metric'].replace('_', ' ')}" for r in rows], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("A minus B, 95% paired bootstrap interval over articles")
    title(ax, name)
    fig.tight_layout()
    fig.savefig(out / f"{path.stem.replace('_metrics', '')}_differences.pdf")
    plt.close(fig)
    return rows


def loop(full, out):
    report = maybe(full / "analysis/loop_metrics.json")
    rounds = {}
    for key, _ in LOOPS:
        path = full / f"runs/loop-{key}/results.jsonl"
        if path.exists():
            lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            if lines:
                rounds[key] = Counter(len(r.get("rounds", [])) for r in lines)
    if not report and not rounds:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    shown = [(k, n) for k, n in LOOPS if report and k in report["means"]]
    colors = ("gray", "orange", "yellow", "purple", "green", "blue", "red")
    if report:
        metric_names = (("cascade_rougeL", "Cascade\nROUGE-L"), ("cascade_recall_shared", "Flagged,\nshared mention"),
                        ("cascade_recall_unshared", "Flagged,\nno shared mention"), ("rougeL", "UpdateROUGE-L"))
        systems = [("copy", "Copy")] + shown
        width = 0.8 / len(systems)
        for i, (key, name) in enumerate(systems):
            v = [report["means"][key].get(m) or 0 for m, _ in metric_names]
            bar(ax1, np.arange(len(metric_names)) + (i - (len(systems) - 1) / 2) * width, v,
                (("gray",) + colors)[i % 8], width=width, label=name, hatch="////" if key == "copy" else None)
        ax1.set_xticks(range(len(metric_names)), [n for _, n in metric_names], fontsize=6.3)
        ax1.set_ylim(0, 1.05)
        ax1.legend(fontsize=5.8, loc="upper right")
        title(ax1, f"(a) Loop runs ({report['articles']} articles)")
    shown_rounds = [(k, n) for k, n in LOOPS if k in rounds]
    top = max((r for c in rounds.values() for r in c), default=1)
    width = 0.8 / max(len(shown_rounds), 1)
    for i, (key, name) in enumerate(shown_rounds):
        total = sum(rounds[key].values())
        v = [rounds[key].get(r, 0) / total for r in range(1, top + 1)]
        bar(ax2, np.arange(1, top + 1) + (i - (len(shown_rounds) - 1) / 2) * width, v, colors[i], width=width,
            label=name)
    ax2.set_xticks(range(1, top + 1))
    ax2.set_xlabel("rounds run")
    ax2.set_ylabel("share of articles")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=5.8)
    title(ax2, "(b) Rounds per article")
    fig.tight_layout()
    fig.savefig(out / "loop.pdf")
    plt.close(fig)
    return {"articles": report and {k: report[k] for k in ("articles", "means", "counted")},
            "rounds": {k: dict(v) for k, v in rounds.items()}}


def newcombe(k1, n1, k2, n2):
    """p1 - p2 and its 95% interval by Newcombe's method from two Wilson intervals (unpaired)."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - np.hypot(p1 - l1, u2 - p2), d + np.hypot(u1 - p1, p2 - l2)


def test_row(report, pair, metric, label):
    """A row from a paired-test table, or a pending row when the pair has not run."""
    tests = report if isinstance(report, list) else (report or {}).get("tests", [])
    for r in tests:
        if r.get("pair") == pair and r.get("metric", metric) == metric:
            d = r.get("diff", r.get("diff_seed42"))
            return {"label": label, "diff": d, "low": r["ci_low"], "high": r["ci_high"], "p": r["p_holm"],
                    "kind": "paired"}
    return {"label": label, "kind": "pending"}


def tested(report, pair, metric, label, fallback):
    """The paired-test row of `pair` when the test has run, else `fallback`."""
    row = test_row(report, pair, metric, label)
    return fallback if row["kind"] == "pending" else row


def point_row(label, a, b):
    if a is None or b is None:
        return {"label": label, "kind": "pending"}
    return {"label": label, "diff": a - b, "kind": "point"}


def ablation_rows(full, numbers):
    get = lambda path, *keys: _dig(maybe(full / path), keys)
    cls_tests = maybe(full / "analysis/classifier_tests.json")
    stale_tests = maybe(full / "analysis/stale_tests.json")
    fix = maybe(full / "analysis/fix_metrics.json")
    ablate = maybe(full / "analysis/ablate_metrics.json")
    loop = maybe(full / "analysis/loop_metrics.json")
    seeds = numbers["seeds"]
    panels = []

    rows = [test_row(cls_tests, "ce24:baseline", None, "LLaDA states vs MiniLM embeddings"),
            test_row(cls_tests, "ce24:noev24", None, "states with vs without evidence"),
            test_row(cls_tests, "ce24:ce32", None, "block 24 vs block 32"),
            test_row(cls_tests, "ce24:focal24", None, "cross-entropy vs focal loss")]
    base = get("heads/classifier/layer24/metrics.json", "test", "macro_f1")
    for pool in ("mean", "nearest", "attention"):
        label = f"+ evidence pooling ({pool}) vs none"
        rows.append(tested(cls_tests, f"{pool}24:ce24", None, label, point_row(
            label, get(f"heads/evidence_{pool}/layer24/metrics.json", "test", "macro_f1"), base)))
    panels.append(("(a) Sentence classifier", "test macro-F1 difference", rows))

    rows = []
    label = "linear: states with vs without evidence"
    fallback = None
    if "stale-bce" in seeds and "noev-stale-bce" in seeds:
        a, b = np.array(seeds["stale-bce"]), np.array(seeds["noev-stale-bce"])
        half = 1.96 * np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        d = a.mean() - b.mean()
        fallback = {"label": label, "diff": d, "low": d - half, "high": d + half, "kind": "seeds"}
    row = tested(stale_tests, "linear:noev", "pr_auc", label, fallback)
    if row:
        rows.append(row)
    if "stale-bce" in seeds and "stale-focal" in seeds:
        a, b = np.array(seeds["stale-bce"]), np.array(seeds["stale-focal"])
        half = 1.96 * np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        d = a.mean() - b.mean()
        rows.append({"label": "linear: BCE vs focal loss", "diff": d, "low": d - half, "high": d + half,
                     "kind": "seeds"})
    att = get("heads/staleness_attention/metrics.json", "test", "pr_auc")
    label = "attention head vs linear head"
    rows.append(tested(stale_tests, "attention:linear", "pr_auc", label, point_row(
        label, att, get("heads/staleness/metrics.json", "test", "pr_auc"))))
    for key, pair, name in (("shuffled_evidence", "attention:shuffled", "shuffled"),
                            ("no_evidence", "attention:none", "removed")):
        label = f"attention: evidence vs {name} at test"
        rows.append(tested(stale_tests, pair, "pr_auc", label, point_row(
            label, att, get("heads/staleness_attention/metrics.json", "controls", key, "pr_auc"))))
    panels.append(("(b) Token staleness heads", "test PR-AUC difference", rows))

    rows = [test_row(fix, "oracle:oracle-noev", "rougeL", "with vs without evidence (UpdateROUGE-L)"),
            test_row(fix, "oracle:oracle-noev", "entity_recall", "with vs without evidence (entity recall)"),
            test_row(fix, "oracle:predicted", "rougeL", "oracle vs linear-head masks"),
            test_row(fix, "oracle:attention", "rougeL", "oracle vs attention-head masks"),
            test_row(fix, "attention:predicted", "rougeL", "attention vs linear-head masks"),
            test_row(fix, "tuned:attention", "rougeL", "val-tuned vs default repair setting"),
            test_row(fix, "oracle:tuned", "rougeL", "oracle masks, default setting vs val-tuned attention masks"),
            test_row(ablate, "oracle:same", "rougeL", "lengths full vs same length only"),
            test_row(ablate, "oracle:pll", "rougeL", "combined reranker vs PLL only"),
            test_row(ablate, "oracle:verifier", "rougeL", "combined reranker vs verifier only")]
    fixabl = maybe(full / "analysis/fixabl_metrics.json")
    rows.append(test_row(fixabl, "oracle-tuned:tuned", "rougeL",
                         "300 articles: oracle vs attention masks, both val-tuned"))
    for key, name in (("penalty", "edit penalty"), ("reject", "unsupported-name rejection"),
                      ("veto", "evidence veto"), ("cap", "span cap"), ("copy", "evidence-copy candidates")):
        rows.append(test_row(fixabl, f"tuned:no-{key}", "rougeL", f"val-tuned vs without the {name}"))
    rows.append(point_row("learned reranker vs oracle pick (sentence ROUGE-L)",
                          get("runs/fix-oracle/metrics.json", "chosen_rougeL"),
                          get("runs/fix-oracle/metrics.json", "pool_rougeL")))
    panels.append(("(c) Evidence-conditioned repair", "test difference", rows))

    rows = []
    det = {k: maybe(full / f"runs/{k}.json") for k, _ in DETECTORS}
    names = dict(DETECTORS)
    if det.get("consequence"):
        c = det["consequence"]
        n = c["group_a_shared_mention"]["cascades"] + c["group_b_no_shared_mention"]["cascades"]
        for key in ("mention", "staleness_delta", "edited_as_evidence"):
            if det.get(key):
                d, lo, hi = newcombe(c["flagged_cascades"], n, det[key]["flagged_cascades"], n)
                rows.append({"label": f"recall: consequence vs {names[key].replace(chr(10), ' ').lower()}",
                             "diff": d, "low": lo, "high": hi, "kind": "unpaired"})
            else:
                rows.append({"label": f"consequence head vs {names[key].replace(chr(10), ' ').lower()}",
                             "kind": "pending"})
    whole = {k: maybe(full / f"runs/all_{k}.json") for k, _ in DETECTORS}
    if whole.get("consequence"):
        c = whole["consequence"]
        for key in ("mention", "staleness_delta", "edited_as_evidence"):
            label = f"precision, all articles: consequence vs {names[key].replace(chr(10), ' ').lower()}"
            if whole.get(key):
                d, lo, hi = newcombe(c["flagged_cascades"], c["flagged"], whole[key]["flagged_cascades"],
                                     whole[key]["flagged"])
                rows.append({"label": label, "diff": d, "low": lo, "high": hi, "kind": "unpaired"})
            else:
                rows.append({"label": label, "kind": "pending"})
    panels.append(("(d) Cascade detection", "test difference (recall on linked articles unless noted)", rows))

    rows = []
    for key, name in (("consequence", "consequence head"), ("edited_as_evidence", "edited as evidence"),
                      ("staleness_delta", "staleness delta")):
        rows.append(test_row(loop, f"{key}:none", "cascade_rougeL", f"loop with {name} vs one round"))
    rows.append(test_row(loop, "consequence:oracle-labels", "rougeL", "predicted vs gold sentence labels"))
    whole = maybe(full / "analysis/all_metrics.json")
    rows += [test_row(whole, "consequence:none", "rougeL", "all articles: loop vs one round (UpdateROUGE-L)"),
             test_row(whole, "consequence:none", "entity_recall", "all articles: loop vs one round (entity recall)"),
             test_row(whole, "consequence:noev", "rougeL", "all articles: loop with vs without evidence"),
             test_row(whole, "consequence:noev", "entity_recall",
                      "all articles: loop with vs without evidence (entity recall)")]
    panels.append(("(e) Iterative loop", "test difference (cascade ROUGE-L unless noted)", rows))

    rows = [test_row(fix, "attention:editor", "rougeL", "supported articles: one round vs T5 editor"),
            test_row(fix, "tuned:editor", "rougeL", "supported articles: val-tuned round vs T5 editor"),
            test_row(fix, "attention:editor", "entity_recall", "supported articles: one round vs T5 editor (entity recall)"),
            test_row(fix, "oracle:editor", "rougeL", "supported articles: oracle masks vs T5 editor"),
            test_row(whole, "consequence:editor", "rougeL", "all articles: full loop vs T5 editor"),
            test_row(whole, "consequence:editor", "edit_distance", "all articles: full loop vs T5 editor (edit distance)")]
    panels.append(("(f) Fine-tuned editor baseline", "test difference (UpdateROUGE-L unless noted)", rows))
    return panels


def _dig(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def ablations(full, numbers, out):
    panels = ablation_rows(full, numbers)
    heights = [max(len(rows), 1) for _, _, rows in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.8, 0.24 * sum(heights) + 0.75 * len(panels)),
                             gridspec_kw={"height_ratios": heights})
    for ax, (name, xlabel, rows) in zip(axes, panels):
        done = [r for r in rows if r["kind"] != "pending"]
        span = max([abs(v) for r in done for v in (r["diff"], r.get("low", 0), r.get("high", 0))] + [0.05])
        for k, r in enumerate(rows):
            if r["kind"] == "pending":
                ax.text(0, k, "not run yet", ha="center", va="center", fontsize=6, color=MUTED,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
                continue
            has_ci = "low" in r
            significant = has_ci and (r["low"] > 0 or r["high"] < 0) and r.get("p", 0) < 0.05
            color = "blue" if significant else "gray"
            if has_ci:
                ax.errorbar(r["diff"], k, xerr=[[r["diff"] - r["low"]], [r["high"] - r["diff"]]], fmt="o", ms=3.4,
                            color=stroke(color), mfc=stroke(color), capsize=1.6, lw=0.9)
            else:
                ax.plot(r["diff"], k, "D", ms=3.4, mfc="white", mec=stroke("orange"), mew=0.9)
            tag = {"paired": f"Holm p {r.get('p', 1):.2g}", "seeds": "5 seeds", "unpaired": "Newcombe",
                   "point": "no test"}[r["kind"]]
            ax.text(1.01, k, f"{r['diff']:+.3f}   {tag}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=6, color=INK if significant else MUTED)
        ax.axvline(0, color=MUTED, lw=0.7)
        ax.set_xlim(-1.15 * span, 1.15 * span)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2g}" if v else "0"))
        ax.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=6.4)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel, fontsize=6.6, labelpad=1)
        title(ax, name)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(out / "ablations.pdf")
    plt.close(fig)
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", required=True, help="output directory of jobs/jl/run_full.sh")
    args = parser.parse_args()
    full = Path(args.full)
    out = full / "analysis/figures"
    out.mkdir(parents=True, exist_ok=True)
    numbers = {"seeds": seed_scores(full)}
    numbers["classifier"] = classifier(full, numbers["seeds"], out)
    numbers["confusion"] = confusion(full, out)
    seeds_figure(numbers["seeds"], out)
    numbers["classifier_tests"] = maybe(full / "analysis/classifier_tests.json")
    numbers["stale_tests"] = maybe(full / "analysis/stale_tests.json")
    numbers["staleness"] = staleness(full, out)
    numbers["consequence_pairs"] = (maybe(full / "heads/consequence/config.json") or {}).get("test")
    numbers["cascades"] = cascades(full, out)
    numbers["infilling"] = infilling(full, out)
    numbers["fix_tests"] = differences(full / "analysis/fix_metrics.json", out, "Single-round infilling on test",
                                       ("rougeL", "entity_precision", "entity_recall", "unsupported"))
    numbers["loop"] = loop(full, out)
    numbers["loop_tests"] = differences(full / "analysis/loop_metrics.json", out, "Loop runs on test",
                                        ("cascade_rougeL", "rougeL", "entity_recall"))
    numbers["ablations"] = ablations(full, numbers, out)
    (full / "analysis/numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(f"figures in {out}: {', '.join(sorted(p.name for p in out.glob('*.pdf')))}")


if __name__ == "__main__":
    main()
