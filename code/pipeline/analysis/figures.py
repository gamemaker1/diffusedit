"""Figures for docs/progress/main.tex, the ACL mid submission.

The figures are the internal report's (analysis/progress_figures.py), with its
palette, font and helpers imported from there. The ACL column is 219 pt
(3.03 in) wide, so most are redrawn at that width with their panels stacked
and are placed with width=\\columnwidth. The two-sentence staleness figure
keeps the internal full-width layout and goes in a figure* float.

    audit.pdf          audited label error rate against the confident-learning estimate
    seeds.pdf          five seeds of the sentence classifier
    stale_scores.pdf   staleness score density of stale and correct tokens
    entities.pdf       entity measures of the infilling runs with 95% intervals, one panel
    update_rouge.pdf   UpdateROUGE of the three infilling runs, both gold sides
    stale_tokens.pdf   per-token scores in two held-out sentences (full width)

Usage:
    python -m analysis.figures --data ../datasets/collated/train.jsonl --datasets ../datasets \\
        --outputs outputs/jl --out ../../docs/progress/figures
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from analysis import progress_figures as pf
from analysis.progress_figures import (BAND, INK, LABELS, MUTED, SEEDS, bar, fill, load, metric_of, plt, ratio_ci,
                                       stroke, title, wilson)

WIDTH = 3.03


def audit(datasets, out):
    """Share of labels judged wrong: the earlier estimate against the reviewers' majority on the random sample."""
    sheets = []
    for path in sorted((datasets / "audit/annotations").glob("*/train_audit.tsv")):
        with open(path, newline="") as f:
            sheets.append(list(csv.DictReader(f, delimiter="\t")))
    n_rows = min(sum(1 for r in sheet if r["verdict"]) for sheet in sheets)
    wrong, total = Counter(), Counter()
    for i in range(n_rows):
        given = sheets[0][i]["label"]
        verdict, votes = Counter(sheet[i]["verdict"] for sheet in sheets).most_common(1)[0]
        if votes * 2 <= len(sheets):
            continue  # no majority
        for key in (given, "all"):
            total[key] += 1
            wrong[key] += verdict != given
    noise = load(datasets / "audit/noise.json")
    names = [*LABELS, "all"]
    x = np.arange(len(names))
    est = [noise["estimated_noise_rate"][c] for c in LABELS] + [noise["estimated_noisy_sentences"] / noise["sentences"]]
    rates = [wilson(wrong[c], total[c]) for c in names]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.6))
    bar(ax, x - 0.2, est, "gray", width=0.38, label="confident-learning estimate, 91,699 sentences")
    bar(ax, x + 0.2, [r[0] for r in rates], "orange", width=0.38,
        yerr=[[r[0] - max(r[1], 0) for r in rates], [r[2] - r[0] for r in rates]],
        error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": INK}, label="hand audit, 100 random sentences")
    for xi, c, e, r in zip(x, names, est, rates):
        ax.text(xi - 0.2, e + 0.015, f"{e:.0%}", ha="center", va="bottom", fontsize=6.2)
        ax.text(xi + 0.2, r[2] + 0.015, f"{r[0]:.0%}\n{wrong[c]}/{total[c]}", ha="center", va="bottom",
                fontsize=6.0, linespacing=0.95)
    ax.set_xticks(x, [f"{c} labels" if c != "all" else "all labels" for c in names], fontsize=6.6)
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("labels judged wrong")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=6.4)
    fig.tight_layout()
    fig.savefig(out / "audit.pdf")
    plt.close(fig)

def seeds(outputs, out):
    """Panel (a) of the internal seeds figure: the sentence classifier over five seeds."""
    sweep = outputs / "heads/sweep"
    fig, ax1 = plt.subplots(figsize=(WIDTH, 1.5))
    configs = [("MiniLM\nbaseline", sweep / "baseline", "gray"),
               ("block 24\ncross-entropy", sweep / "cls-ce/layer24", "blue"),
               ("block 24\nfocal loss", sweep / "cls-focal/layer24", "orange"),
               ("block 32\ncross-entropy", sweep / "cls-ce/layer32", "blue"),
               ("block 32\nfocal loss", sweep / "cls-focal/layer32", "orange")]
    rng = np.random.default_rng(1)
    values = []
    for k, (name, path, color) in enumerate(configs):
        v = np.array([metric_of(path / f"seed{s}" / "metrics.json", "macro_f1") for s in SEEDS])
        values.append(v)
        ax1.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=fill(color), edgecolor=stroke(color),
                    linewidth=0.7, zorder=3)
        ax1.plot([k - 0.28, k + 0.28], [v.mean()] * 2, color=stroke(color), lw=1.6)
        ax1.text(k, v.max() + 0.005, f"{v.mean():.3f}\n$\\pm${v.std(ddof=1):.3f}", ha="center", fontsize=6.0)
    ax1.set_xticks(range(len(configs)), [c[0] for c in configs], fontsize=6.0)
    ax1.set_ylabel("held-out macro-F1")
    ax1.grid(axis="x", visible=False)
    lo, hi = min(v.min() for v in values), max(v.max() for v in values)
    ax1.set_ylim(lo - 0.015, hi + 0.035)
    fig.tight_layout()
    fig.savefig(out / "seeds.pdf")
    plt.close(fig)

def stale_scores(plots, out):
    """Panel (a) of the internal figure: score density of stale and correct tokens."""
    a = load(plots / "stale_analysis.json")
    fig, ax = plt.subplots(figsize=(WIDTH, 1.45))
    bins = np.array(a["hist"]["bins"])
    centers, width = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
    curves = {}
    for key, color, name in (("fresh", "blue", "correct tokens"), ("stale", "orange", "stale tokens")):
        v = np.array(a["hist"][key], float)
        curves[key] = v / v.sum() / width
        ax.fill_between(centers, curves[key], color=fill(color), alpha=0.6, lw=0)
        ax.plot(centers, curves[key], color=stroke(color), lw=1.4, label=f"{name} ({int(v.sum()):,})")
    with plt.rc_context({"hatch.color": MUTED, "hatch.linewidth": 0.4}):
        ax.fill_between(centers, np.minimum(curves["fresh"], curves["stale"]), facecolor="none",
                        edgecolor=MUTED, hatch="////", lw=0)
    ax.text(0.5, 0.75, "overlap", ha="center", fontsize=6.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.6)
    ax.set_xlabel("staleness score")
    ax.set_ylabel("density")
    ax.text(0.99, 2.45, f"AUROC {a['auc_global']:.3f}\n(0.5 = random)", ha="right", va="top", fontsize=6.6)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.0),
               fontsize=6.3, handlelength=1.2, columnspacing=1.0)
    fig.savefig(out / "stale_scores.pdf")
    plt.close(fig)

def entities(outputs, data, out):
    """The internal entity figure in one panel. The unchanged sentence is a reference tick, not a bar."""
    from infilling.significance import collect
    runs = {"oracle": outputs / "runs/fix-oracle", "noev": outputs / "runs/fix-oracle-noev",
            "predicted": outputs / "runs/fix-predicted"}
    _, _, sums = collect(data, runs)
    systems = [("oracle", "oracle masks", "blue"), ("noev", "oracle masks, no evidence", "orange"),
               ("predicted", "predicted masks", "green")]
    metrics = [("precision", "precision"), ("recall", "recall"), ("new_recall", "new-entity\nrecall"),
               ("stale_kept", "outdated\nkept"), ("unsupported", "fabricated")]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(WIDTH, 1.7))
    width = 0.26
    ax.axvspan(2.5, 4.5, color=BAND, zorder=0)
    ax.text(1.0, 1.1, "higher is better", ha="center", fontsize=6.2, color=MUTED)
    ax.text(3.5, 1.1, "lower is better", ha="center", fontsize=6.2, color=MUTED)
    for i, (system, name, color) in enumerate(systems):
        v, lo, hi = [], [], []
        for metric, _ in metrics:
            num, den = sums["all"][system][metric]
            c, low, high = ratio_ci(num, den, rng)
            v.append(c), lo.append(c - low), hi.append(high - c)
        bar(ax, np.arange(len(metrics)) + (i - 1) * width, v, color, width=width * 0.92, label=name, yerr=[lo, hi],
            error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": INK})
    for k, (metric, _) in enumerate(metrics):
        num, den = sums["all"]["copy"][metric]
        c = num.sum() / den.sum()
        ax.plot([k - 0.45, k + 0.45], [c, c], color=INK, lw=1.0, ls="--", zorder=5,
                label="unchanged sentence" if k == 0 else None)
    ax.set_xticks(np.arange(len(metrics)), [label for _, label in metrics], fontsize=6.4)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("share of entities")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(n) for n in ("oracle masks", "oracle masks, no evidence", "predicted masks",
                                       "unchanged sentence")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=2,
               bbox_to_anchor=(0.55, 1.0), columnspacing=0.8, fontsize=6.4, handlelength=1.6)
    fig.savefig(out / "entities.pdf")
    plt.close(fig)

def update_rouge(outputs, out):
    """The internal UpdateROUGE figure, both gold sides side by side."""
    report = load(outputs / "runs/update_rouge.json")
    systems = [("oracle", "oracle masks", "blue"), ("noev", "oracle masks, no evidence", "orange"),
               ("predicted", "predicted masks", "green")]
    metrics = ("rouge1", "rouge2", "rougeL")
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.6), sharey=True)
    for ax, (side, head) in zip(axes, (("full", "(a) all changed gold\nsentences"),
                                       ("edits", "(b) gold rewrites of\nEDIT sentences only"))):
        per = report[side]["per_article"]
        for i, (name, label, color) in enumerate(systems):
            v, lo, hi = [], [], []
            for m in metrics:
                s = np.array(per[name][m])
                c, low, high = ratio_ci(s, np.ones(len(s)), rng)
                v.append(c), lo.append(c - low), hi.append(high - c)
            bar(ax, np.arange(3) + (i - 1) * 0.26, v, color, width=0.24, label=label, yerr=[lo, hi],
                error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": INK})
        ax.set_xticks(range(3), ["R-1", "R-2", "R-L"], fontsize=6.6)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="x", visible=False)
        ax.set_title(head, loc="left", fontsize=6.8, color=INK)
    axes[0].set_ylabel("UpdateROUGE F1")
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.86))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.0), fontsize=6.0,
               columnspacing=0.7, handlelength=1.1, handletextpad=0.4)
    fig.savefig(out / "update_rouge.pdf")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="collated train.jsonl, for the entity measures")
    parser.add_argument("--datasets", required=True, help="the code/datasets folder, for the audit")
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    outputs, out, datasets = Path(args.outputs), Path(args.out), Path(args.datasets)
    out.mkdir(parents=True, exist_ok=True)
    audit(datasets, out)
    seeds(outputs, out)
    stale_scores(outputs / "plots", out)
    entities(outputs, args.data, out)
    update_rouge(outputs, out)
    pf.stale_tokens(outputs / "plots", out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
