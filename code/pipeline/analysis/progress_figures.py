"""Figures for docs/internal/progress/progress.tex, built from saved outputs.

Every figure reads saved run outputs or the collated data. Nothing here uses
the GPU. Text is set in Computer Modern Sans (cmss10, shipped with
matplotlib), the face of the report's headings and captions, on a pastel
palette with a darker stroke of each hue for edges and lines.

    data_build.pdf     build filters per FRUIT source, and labels before and
                       after the KEEP fix
    aligner.pdf        best-match similarity of source sentences by label
    label_noise.pdf    confident-learning joint and the hand review
    classifier.pdf     sentence classifiers and evidence inputs
    seeds.pdf          five seeds of each loss for both heads
    stale_scores.pdf   staleness score distribution and share stale per bin
    stale_rules.pdf    threshold sweep and precision against recall
    stale_tokens.pdf   per-token scores in two held-out sentences
    entities.pdf       entity measures of the four infilling systems
    update_rouge.pdf   UpdateROUGE of the three infilling runs
    differences.pdf    paired differences with 95% intervals

Usage:
    python -m analysis.progress_figures --data ../datasets/collated/train.jsonl \\
        --datasets ../datasets --outputs outputs/jl --out ../../docs/internal/progress/figures
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# cmss10 has no minus sign, so signed tick labels go through mathtext
SIGNED = FuncFormatter(lambda v, _: f"${v:g}$")

from infilling.significance import collect  # noqa: E402

# pastel fill and darker stroke of the same hue
PASTEL = {
    "blue": ("#AFCBEA", "#3F6E9E"),
    "orange": ("#F7C59F", "#B9652A"),
    "green": ("#B7DEC3", "#3E8A5A"),
    "purple": ("#D3C4EA", "#6B529C"),
    "yellow": ("#F3DFA2", "#A88420"),
    "gray": ("#D7DADF", "#59626D"),
    "red": ("#F2B8B5", "#A8423E"),
}
INK, MUTED, GRID, BAND = "#1C2B33", "#5B6770", "#E3E8EC", "#F1F4F7"
LABELS = ("KEEP", "EDIT", "DROP")
LABEL_COLOR = {"KEEP": "blue", "EDIT": "orange", "DROP": "purple"}
SEEDS = (42, 1, 2, 3, 4)

plt.rcParams.update({
    "font.family": "cmss10", "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def fill(name):
    return PASTEL[name][0]


def stroke(name):
    return PASTEL[name][1]


def bar(ax, x, h, name, **kw):
    return ax.bar(x, h, color=fill(name), edgecolor=stroke(name), linewidth=0.8, **kw)


def load(path):
    return json.loads(Path(path).read_text())


def title(ax, text):
    ax.set_title(text, loc="left", color=INK)


def note(ax, xy, text, xytext, ha="left", **kw):
    """An annotation in ink with a thin muted arrow from xytext to xy, both in data units."""
    ax.annotate(text, xy=xy, xytext=xytext, textcoords="data", fontsize=6.6, color=INK, ha=ha,
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.6, "shrinkA": 1, "shrinkB": 2,
                            "mutation_scale": 6}, **kw)


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, centre - half, centre + half


def metric_of(path, key):
    m = load(path)
    return (m.get("heldout") or m.get("val"))[key] if key == "macro_f1" else m[key]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def data_build(datasets, out):
    tallies = load(datasets / "archive/collated/tallies.json")
    sources = [("train", "train (shards 0 to 8)"), ("dev", "dev (shard 9)"), ("test", "gold test")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.55, 1]})
    parts = [("kept", "kept", "green"), ("append_only", "only additions, no EDIT or DROP", "yellow"),
             ("article_over_budget", "article over 400 tokens", "red")]
    for k, (src, name) in enumerate(sources):
        t = tallies[src]
        total = sum(t.values())
        left = 0
        for key, label, color in parts:
            share = t[key] / total
            ax1.barh(k, share, left=left, height=0.62, color=fill(color), edgecolor=stroke(color), linewidth=0.8,
                     label=label if k == 0 else None)
            ax1.text(left + share / 2, k, f"{t[key]:,}", ha="center", va="center", fontsize=6.5)
            left += share
        ax1.text(1.015, k, f"{total:,}\nread", va="center", fontsize=6.4, color=MUTED)
    ax1.set_yticks(range(len(sources)), [name for _, name in sources])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("share of FRUIT instances")
    ax1.grid(axis="y", visible=False)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, columnspacing=0.8, handlelength=1.2,
               fontsize=6.8)
    title(ax1, "(a) Instances kept by the build filters")

    before, after = Counter(), Counter()
    for line in open(datasets / "archive/collated/train.pre_keepfix.jsonl"):
        before.update(lab["label"] for lab in json.loads(line)["labels"])
    for line in open(datasets / "collated/train.jsonl"):
        after.update(lab["label"] for lab in json.loads(line)["labels"])
    x = np.arange(3)
    bar(ax2, x - 0.2, [before[c] / 1000 for c in LABELS], "gray", width=0.38)
    for i, c in enumerate(LABELS):
        bar(ax2, x[i] + 0.2, after[c] / 1000, LABEL_COLOR[c], width=0.38)
    moved = before["KEEP"] - after["KEEP"]
    note(ax2, (0.2, after["KEEP"] / 1000 + 2), f"{moved:,} sentences moved\nfrom KEEP to EDIT", (0.95, 200))
    ax2.set_xticks(x, LABELS)
    ax2.set_ylabel("train sentences (thousands)")
    ax2.set_ylim(0, 225)
    ax2.grid(axis="x", visible=False)
    ax2.legend(handles=[Patch(facecolor=fill("gray"), edgecolor=stroke("gray"), label="before the fix"),
                        Patch(facecolor="white", edgecolor=MUTED, label="after the fix")],
               loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.8)
    title(ax2, "(b) Train labels, 84,117 articles")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(out / "data_build.pdf")
    plt.close(fig)


def aligner(datasets, out, sample=10000, seed=0):
    sys.path.insert(0, str(datasets))
    sys.path.insert(0, str(datasets / "fruit"))
    import collate
    lines = open(datasets / "collated/train.jsonl").readlines()
    best = {c: [] for c in LABELS}
    for line in random.Random(seed).sample(lines, sample):
        r = json.loads(line)
        src, tgt = r["source_sentences"], [t["text"] for t in r["target_sentences"]]
        _, sim = collate.align(src, tgt)
        for i, lab in enumerate(r["labels"]):
            best[lab["label"]].append(max(sim[i]) if sim[i] else 0.0)
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    bins = np.linspace(0, 1, 41)
    bottom = np.zeros(40)
    for c in ("DROP", "EDIT", "KEEP"):
        h, _ = np.histogram(best[c], bins)
        ax.bar(bins[:-1], h, width=bins[1] - bins[0], bottom=bottom, align="edge", color=fill(LABEL_COLOR[c]),
               edgecolor=stroke(LABEL_COLOR[c]), linewidth=0.5, label=f"{c} ({len(best[c]):,} sentences)")
        bottom += h
    ax.set_yscale("log")
    top = bottom.max() * 80
    ax.set_ylim(3, top)
    ax.axvspan(0, 0.35, color=BAND, zorder=0)
    for x0 in (0.35, 0.95):
        ax.axvline(x0, color=INK, lw=0.8, ls="--")
    ax.text(0.34, top / 1.6, "below 0.35: no partner,\nthe sentence is DROP", ha="right", va="top", fontsize=6.6)
    ax.text(0.94, top / 1.6, "0.95 and above: KEEP, unless\na name or number changed", ha="right", va="top",
            fontsize=6.6)
    ax.text(0.6, top / 1.6, "between: EDIT", ha="center", va="top", fontsize=6.6)
    n_drop_above = sum(v >= 0.35 for v in best["DROP"])
    ax.text(0.37, top / 45, f"{n_drop_above:,} DROP sentences sit above 0.35: their best\npartner went to "
            "another source sentence with\na higher similarity", fontsize=6.4, color=stroke("purple"))
    n_edit_above = sum(v >= 0.95 for v in best["EDIT"])
    note(ax, (0.975, 150), f"{n_edit_above:,} EDIT sentences at 0.95 or\nabove changed a name or number",
         (0.80, top / 45), ha="center")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("similarity to the best-matching target sentence (F1 overlap of word multisets)")
    ax.set_ylabel("source sentences (log scale)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.74), fontsize=6.8)
    title(ax, f"How the aligner labels source sentences, {sample:,} random train articles")
    fig.tight_layout()
    fig.savefig(out / "aligner.pdf")
    plt.close(fig)


def label_noise(datasets, out):
    noise = load(datasets / "audit/noise.json")
    joint = noise["confident_joint"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw={"width_ratios": [1, 1.3]})
    m = np.array([[joint[g][t] for t in LABELS] for g in LABELS])
    share = m / m.sum(1, keepdims=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("pastel", ["#FFFFFF", fill("blue"), "#7FA6CF"])
    ax1.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f"{share[i, j]:.0%}\n{m[i, j]:,.0f}", ha="center", va="center", fontsize=6.8,
                     fontweight="bold" if i == j else "normal")
    ax1.set_xticks(range(3), LABELS)
    ax1.set_yticks(range(3), LABELS)
    ax1.set_xlabel("label the classifier points to")
    ax1.set_ylabel("label given by the aligner")
    ax1.grid(False)
    for s in ax1.spines.values():
        s.set_visible(False)
    title(ax1, "(a) Confident-learning joint")

    review = {"KEEP": (65, 8), "EDIT": (33, 13), "DROP": (13, 3), "all": (111, 24)}
    names = [*LABELS, "all"]
    x = np.arange(len(names))
    est = [noise["estimated_noise_rate"][c] for c in LABELS] + \
          [noise["estimated_noisy_sentences"] / noise["sentences"]]
    bar(ax2, x - 0.2, est, "gray", width=0.38, label="confident-learning estimate, 91,699 sentences")
    rates = [wilson(review[c][1], review[c][0]) for c in names]
    bar(ax2, x + 0.2, [r[0] for r in rates], "orange", width=0.38,
        yerr=[[r[0] - r[1] for r in rates], [r[2] - r[0] for r in rates]],
        error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": INK}, label="hand review of 111 flagged sentences")
    for xi, c, r in zip(x, names, rates):
        n, k = review[c]
        ax2.text(xi + 0.2, r[2] + 0.02, f"{k}/{n}", ha="center", fontsize=6.2, va="bottom")
    ax2.set_xticks(x, ["given\nKEEP", "given\nEDIT", "given\nDROP", "all"])
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("share of labels judged wrong")
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="upper left", fontsize=6.6)
    ax2.text(3.55, 0.70, "The flagged sentences are the 300 the\nclassifier disputes most. A random\n"
             "sample would show fewer errors.", ha="right", fontsize=6.3, color=MUTED)
    title(ax2, "(b) Estimated and hand-checked error rates")
    fig.tight_layout(w_pad=2)
    fig.savefig(out / "label_noise.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Sentence classifier
# --------------------------------------------------------------------------

def classifier(outputs, out):
    heads = outputs / "heads"
    systems = [("MiniLM baseline (no LLaDA)", heads / "baseline", "gray"),
               ("LLaDA block 24", heads / "classifier/layer24", "blue"),
               ("LLaDA block 32", heads / "classifier/layer32", "green")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={"width_ratios": [1.35, 1]})
    width = 0.26
    macro = {}
    for i, (name, path, color) in enumerate(systems):
        v = load(path / "metrics.json")
        v = v.get("heldout") or v["val"]
        values = [v[c]["f1"] for c in LABELS] + [v["macro_f1"]]
        macro[name] = v["macro_f1"]
        bar(ax1, np.arange(4) + (i - 1) * width, values, color, width=width * 0.92, label=name)
    gain = macro["LLaDA block 24"] - macro["MiniLM baseline (no LLaDA)"]
    note(ax1, (3, macro["LLaDA block 24"] + 0.015), f"LLaDA: +{gain:.3f}\nmacro-F1", (3.0, 0.68), ha="center")
    note(ax1, (2, 0.34), "DROP is the\nhardest class", (1.6, 0.70), ha="center")
    ax1.set_xticks(np.arange(4), [*LABELS, "Macro-F1"])
    ax1.set_ylim(0, 0.82)
    ax1.set_ylabel("F1 on 38,267 held-out sentences")
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, columnspacing=0.8, handlelength=1.2,
               fontsize=6.8)
    title(ax1, "(a) Per-class F1, sentence vector only")

    variants = [("none", "classifier"), ("mean", "evidence"), ("nearest", "evidence_nearest"),
                ("attention", "evidence_attention")]
    for k, (layer, color, marker) in enumerate(((24, "blue", "o"), (32, "green", "s"))):
        values = [metric_of(heads / d / f"layer{layer}" / "metrics.json", "macro_f1") for _, d in variants]
        y = np.arange(4)[::-1] + (0.14 if k == 0 else -0.14)
        ax2.plot(values, y, marker, color=fill(color), mec=stroke(color), mew=0.9, ms=6, label=f"block {layer}")
        for xv, yv in zip(values, y):
            ax2.text(xv + 0.0013, yv, f"{xv:.3f}", va="center", fontsize=6.4)
    ax2.axhspan(2.5, 3.5, color=BAND, zorder=0)
    ax2.text(0.4665, 3.4, "no evidence vector:\nhighest at both blocks", fontsize=6.2, color=MUTED, ha="left",
             va="top")
    ax2.set_yticks(np.arange(4)[::-1], [name for name, _ in variants])
    ax2.set_xlim(0.465, 0.512)
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_xlabel("macro-F1 (axis starts at 0.465)")
    ax2.set_ylabel("evidence vector added")
    ax2.grid(axis="y", visible=False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    title(ax2, "(b) Adding an evidence vector")
    fig.tight_layout(w_pad=2)
    fig.savefig(out / "classifier.pdf")
    plt.close(fig)


def seeds(outputs, out):
    sweep = outputs / "heads/sweep"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    configs = [("baseline", "MiniLM\nbaseline", sweep / "baseline", "gray"),
               ("ce24", "block 24\ncross-entropy", sweep / "cls-ce/layer24", "blue"),
               ("focal24", "block 24\nfocal", sweep / "cls-focal/layer24", "orange"),
               ("ce32", "block 32\ncross-entropy", sweep / "cls-ce/layer32", "blue"),
               ("focal32", "block 32\nfocal", sweep / "cls-focal/layer32", "orange")]
    rng = np.random.default_rng(1)
    for k, (_, name, path, color) in enumerate(configs):
        v = np.array([metric_of(path / f"seed{s}" / "metrics.json", "macro_f1") for s in SEEDS])
        ax1.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=16, color=fill(color), edgecolor=stroke(color),
                    linewidth=0.7, zorder=3)
        ax1.plot([k - 0.28, k + 0.28], [v.mean()] * 2, color=stroke(color), lw=1.6)
        ax1.text(k, v.max() + 0.006, f"{v.mean():.3f}\n$\\pm${v.std(ddof=1):.3f}", ha="center", fontsize=6.0)
    ax1.set_xticks(range(len(configs)), [c[1] for c in configs], fontsize=6.4)
    ax1.set_ylabel("held-out macro-F1")
    ax1.grid(axis="x", visible=False)
    lo = min(metric_of(c[2] / f"seed{s}" / "metrics.json", "macro_f1") for c in configs for s in SEEDS)
    hi = max(metric_of(c[2] / f"seed{s}" / "metrics.json", "macro_f1") for c in configs for s in SEEDS)
    ax1.set_ylim(lo - 0.02, hi + 0.03)
    title(ax1, "(a) Sentence classifier, 5 seeds each")

    for ax, key, name in ((ax2, "val_auroc", "AUROC"), (ax3, "val_pr_auc", "PR-AUC")):
        for k, (loss, label, color) in enumerate((("bce", "binary\ncross-entropy", "blue"), ("focal", "focal", "orange"))):
            v = np.array([load(sweep / f"stale-{loss}/seed{s}/metrics.json")[key] for s in SEEDS])
            ax.scatter(k + rng.uniform(-0.1, 0.1, len(v)), v, s=16, color=fill(color), edgecolor=stroke(color),
                       linewidth=0.7, zorder=3)
            ax.plot([k - 0.25, k + 0.25], [v.mean()] * 2, color=stroke(color), lw=1.6)
            ax.text(k, v.max() + 0.002, f"{v.mean():.3f}\n$\\pm${v.std(ddof=1):.3f}", ha="center", fontsize=6.0)
        ax.set_xticks([0, 1], ["binary\ncross-entropy", "focal"], fontsize=6.4)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(f"held-out {name}")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.35)
        title(ax, f"({'b' if ax is ax2 else 'c'}) Staleness {name}")
    fig.text(0.5, -0.03, "Each dot is one seed (held-out split, initialization and batch order). Bars mark the mean, "
             "labels give mean $\\pm$ standard deviation.", ha="center", fontsize=6.4, color=MUTED)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "seeds.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Staleness head
# --------------------------------------------------------------------------

def stale_scores(plots, out):
    a = load(plots / "stale_analysis.json")
    s = load(plots / "stale_scatter.json")
    fig = plt.figure(figsize=(6.8, 2.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.12, wspace=0.3)
    ax1 = fig.add_subplot(grid[:, 0])
    bins = np.array(a["hist"]["bins"])
    centers, width = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
    curves = {}
    for key, color, name in (("fresh", "blue", "correct tokens"), ("stale", "orange", "stale tokens")):
        v = np.array(a["hist"][key], float)
        curves[key] = v / v.sum() / width
        ax1.fill_between(centers, curves[key], color=fill(color), alpha=0.6, lw=0)
        ax1.plot(centers, curves[key], color=stroke(color), lw=1.4, label=f"{name} (n = {int(v.sum()):,})")
    with plt.rc_context({"hatch.color": MUTED, "hatch.linewidth": 0.4}):
        ax1.fill_between(centers, np.minimum(curves["fresh"], curves["stale"]), facecolor="none",
                         edgecolor=MUTED, hatch="////", lw=0)
    ax1.text(0.5, 0.75, "overlap", ha="center", fontsize=6.8)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 3.0)
    ax1.set_xlabel("staleness score")
    ax1.set_ylabel("density")
    ax1.legend(loc="upper left", fontsize=6.8)
    title(ax1, f"(a) Scores by label (AUROC {a['auc_global']:.3f})")

    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 1], sharex=ax2)
    b = s["binned"]
    mid = np.array([(r["lo"] + r["hi"]) / 2 for r in b])
    n = np.array([r["n"] for r in b])
    st = np.array([r["stale"] for r in b])
    ok = n >= 50
    ax2.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    ax2.plot(mid[ok], st[ok] / n[ok], color=stroke("orange"), lw=1.4, marker="o", ms=2.6, mfc=fill("orange"))
    ax2.text(0.83, 0.93, "calibrated\n(share = score)", fontsize=6.3, color=MUTED, ha="right", va="top")
    ax2.text(0.99, 0.70, "observed", fontsize=6.3, color=stroke("orange"), ha="right")
    ax2.axhline(0.5, color=INK, lw=0.6, ls=":")
    note(ax2, (0.6125, 0.352), "score 0.6: only\n35% are stale", (0.72, 0.1))
    note(ax2, (0.81, 0.5), "stale tokens are the majority\nonly above 0.81", (0.04, 0.62))
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("share stale")
    plt.setp(ax2.get_xticklabels(), visible=False)
    title(ax2, "(b) Share of stale tokens per score bin")
    bar(ax3, mid, n / n.sum() * 100, "gray", width=0.022)
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("staleness score (bins of 0.025)")
    ax3.set_ylabel("% tokens")
    ax3.grid(axis="x", visible=False)
    fig.savefig(out / "stale_scores.pdf")
    plt.close(fig)


def stale_rules(plots, out):
    a = load(plots / "stale_analysis.json")
    curve = [r for r in a["curve"] if r["masked"] > 0]
    tau = [r["tau"] for r in curve]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9))
    ax1.plot(tau, [r["precision"] for r in curve], color=stroke("blue"), lw=1.6, label="precision")
    ax1.plot(tau, [r["recall"] for r in curve], color=stroke("orange"), lw=1.6, label="recall")
    ax1.plot(tau, [r["masked"] for r in curve], color=stroke("green"), lw=1.6, ls="--", label="share masked")
    ax1.axhline(a["base_rate"], color=MUTED, lw=0.6, ls=":")
    ax1.text(0.02, a["base_rate"] - 0.065, f"share stale {a['base_rate']:.3f}", fontsize=6.4, color=MUTED)
    rules = a["rules"]
    for t, key, xt, yt in ((0.4, "global tau 0.40 (trained)", 0.03, 0.8),
                           (0.6, "global tau 0.60 (tau_match)", 0.62, 0.62)):
        r = rules[key]
        ax1.axvline(t, color=INK, lw=0.6, ls="-.")
        ax1.text(xt, yt, f"$\\tau$ = {t}\nmasks {r['masked']:.0%}\nprecision {r['precision']:.2f}\n"
                 f"recall {r['recall']:.2f}", fontsize=6.4,
                 bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": GRID})
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("threshold $\\tau$: mask every token with score $\\geq \\tau$")
    ax1.set_ylabel("value on held-out tokens")
    ax1.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2), columnspacing=0.8, handlelength=1.5)
    title(ax1, "(a) Moving one global threshold")

    oracle = rules["per-sentence top-k, k = true stale count (oracle count)"]
    ax2.axhspan(oracle["precision"], 1, color=BAND, zorder=0)
    ax2.text(0.98, 0.95, "no threshold rule on these scores\nreaches this region", ha="right", va="top",
             fontsize=6.4, color=MUTED)
    ax2.plot([r["recall"] for r in curve], [r["precision"] for r in curve], color=stroke("blue"), lw=1.6,
             label="global threshold, every $\\tau$")
    points = [("global tau 0.40 (trained)", "o", "gray", "$\\tau$ = 0.40", (0.74, 0.12)),
              ("global tau 0.60 (tau_match)", "o", "gray", "$\\tau$ = 0.60", (0.04, 0.2)),
              ("per-sentence top 28.5%", "s", "yellow", "top 28.5% of each sentence", (0.30, 0.05)),
              ("per-sentence top-k, k = true stale count (oracle count)", "D", "orange",
               "top k, k = true stale count\n(upper bound)", (0.62, 0.66))]
    for key, marker, color, name, xytext in points:
        r = rules[key]
        ax2.plot(r["recall"], r["precision"], marker, color=fill(color), mec=stroke(color), mew=0.9, ms=6, zorder=4)
        note(ax2, (r["recall"], r["precision"]), f"{name}\nP {r['precision']:.3f}, R {r['recall']:.3f}", xytext)
    ax2.axhline(a["base_rate"], color=MUTED, lw=0.6, ls=":")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
    title(ax2, "(b) Precision against recall")
    fig.tight_layout(w_pad=2)
    fig.savefig(out / "stale_rules.pdf")
    plt.close(fig)


def stale_tokens(plots, out):
    a = load(plots / "stale_analysis.json")
    chosen = [e for e in a["examples"] if "Fingal" in "".join(e["tokens"])] + \
             [e for e in a["examples"] if "Superliga" in "".join(e["tokens"])]
    texts = ["9 correct tokens score above the one stale token, Mayor",
             "the correct token after outscores all four stale tokens"]
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 3.7))
    for ax, e, text in zip(axes, chosen, texts):
        scores, stale = np.array(e["scores"]), np.array(e["stale"])
        for x, (sc, st) in enumerate(zip(scores, stale)):
            color = "orange" if st else "blue"
            ax.bar(x, sc, width=0.8, color=fill(color), edgecolor=stroke(color), linewidth=0.7)
        for t, ls in ((0.4, "--"), (0.6, "-.")):
            ax.axhline(t, color=INK, lw=0.6, ls=ls)
        ax.set_xticks(range(len(scores)), [t.strip() or "_" for t in e["tokens"]], rotation=55, ha="right",
                      fontsize=6.5)
        ax.set_xlim(-0.6, len(scores) - 0.4)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("staleness score")
        ax.grid(axis="x", visible=False)
        target = e["target"] if len(e["target"]) < 118 else e["target"][:115] + "..."
        ax.set_title(f"gold rewrite: {target}", loc="left", fontsize=6.8, color=MUTED)
        ax.text(len(scores) - 0.5, 0.95, text, ha="right", va="center", fontsize=6.6,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": GRID})
    handles = [Patch(facecolor=fill("orange"), edgecolor=stroke("orange"), label="stale (derived label)"),
               Patch(facecolor=fill("blue"), edgecolor=stroke("blue"), label="correct"),
               plt.Line2D([], [], color=INK, ls="--", lw=0.6, label="$\\tau$ = 0.4"),
               plt.Line2D([], [], color=INK, ls="-.", lw=0.6, label="$\\tau$ = 0.6")]
    fig.tight_layout(h_pad=0.7, rect=(0, 0, 1, 0.94))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(out / "stale_tokens.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Infilling
# --------------------------------------------------------------------------

SYSTEMS = [("copy", "unchanged sentence", "gray"), ("oracle", "oracle masks", "blue"),
           ("noev", "oracle masks, no evidence", "orange"), ("predicted", "predicted masks, $\\tau$ = 0.6", "green")]


def ratio_ci(num, den, rng, resamples=10000):
    idx = rng.integers(0, len(num), size=(resamples, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), *np.nanpercentile(boot, [2.5, 97.5])


def entities(sums, n_articles, n_sentences, out):
    rng = np.random.default_rng(0)
    panels = [("(a) Higher is better", [("precision", "entity\nprecision"), ("recall", "entity\nrecall"),
                                        ("new_recall", "new-entity\nrecall")]),
              ("(b) Lower is better", [("stale_kept", "outdated\nentities kept"),
                                       ("unsupported", "fabricated\nentities")])]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [3, 2]})
    width = 0.2
    values = {}
    for ax, (head, metrics) in zip(axes, panels):
        for i, (system, name, color) in enumerate(SYSTEMS):
            v, lo, hi = [], [], []
            for metric, _ in metrics:
                num, den = sums["all"][system][metric]
                c, low, high = ratio_ci(num, den, rng)
                v.append(c), lo.append(c - low), hi.append(high - c)
                values[(system, metric)] = c
            x = np.arange(len(metrics)) + (i - 1.5) * width
            bar(ax, x, v, color, width=width * 0.92, label=name, yerr=[lo, hi],
                error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": INK})
            for xv, val in zip(x, v):
                if val < 0.005:
                    ax.text(xv, 0.015, "0", ha="center", va="bottom", fontsize=6.4)
        ax.set_xticks(np.arange(len(metrics)), [label for _, label in metrics])
        ax.set_ylim(0, 1.15)
        ax.grid(axis="x", visible=False)
        title(ax, head)
    a0, a1 = axes
    note(a0, (2 - 0.5 * width, values[("oracle", "new_recall")] + 0.13),
         f"evidence raises new-entity\nrecall from {values[('noev', 'new_recall')]:.3f} to "
         f"{values[('oracle', 'new_recall')]:.3f}", (2.0, 0.62), ha="center")
    note(a0, (1 + 1.5 * width, values[("predicted", "recall")] + 0.08),
         "predicted masks fall below\nthe unchanged sentence", (1.0, 0.98), ha="center")
    note(a1, (1 - 0.5 * width, values[("noev", "unsupported")] + 0.06),
         f"without evidence,\n{values[('noev', 'unsupported')]:.0%} of entities\nare fabricated", (0.55, 0.62),
         ha="center")
    a0.set_ylabel("share, summed over sentences")
    fig.tight_layout(w_pad=2, rect=(0, 0, 1, 0.88))
    handles, labels = a0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
    fig.text(0.5, -0.02, f"{n_sentences} repaired sentences in {n_articles} articles. Error bars: 95% bootstrap "
             "intervals over articles.", ha="center", fontsize=6.4, color=MUTED)
    fig.savefig(out / "entities.pdf")
    plt.close(fig)


def update_rouge(outputs, out):
    report = load(outputs / "runs/update_rouge.json")
    systems = [("oracle", "oracle masks", "blue"), ("noev", "oracle masks, no evidence", "orange"),
               ("predicted", "predicted masks, $\\tau$ = 0.6", "green")]
    metrics = [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    for ax, (side, head) in zip(axes, (("full", "(a) Gold side: every changed gold sentence"),
                                       ("edits", "(b) Gold side: rewrites of EDIT sentences"))):
        per = report[side]["per_article"]
        for i, (name, label, color) in enumerate(systems):
            v, lo, hi = [], [], []
            for m, _ in metrics:
                s = np.array(per[name][m])
                c, low, high = ratio_ci(s, np.ones(len(s)), rng)
                v.append(c), lo.append(c - low), hi.append(high - c)
            bar(ax, np.arange(3) + (i - 1) * 0.26, v, color, width=0.24, label=label, yerr=[lo, hi],
                error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": INK})
        ax.set_xticks(range(3), [name for _, name in metrics])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="x", visible=False)
        title(ax, head)
    axes[0].set_ylabel("UpdateROUGE F1, mean over articles")
    axes[1].text(2.45, 0.93, "The unchanged article scores 0:\nit changes no sentence.", ha="right", va="top",
                 fontsize=6.3, color=MUTED)
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.88))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.02, f"{report['articles']['full']} articles. Error bars: 95% bootstrap intervals over articles.",
             ha="center", fontsize=6.4, color=MUTED)
    fig.savefig(out / "update_rouge.pdf")
    plt.close(fig)


def differences(outputs, out):
    rows = load(outputs / "runs/significance.json")["tables"]["all"]
    pairs = [("oracle:copy", "oracle $-$ unchanged"), ("oracle:noev", "oracle $-$ no evidence"),
             ("predicted:copy", "predicted $-$ unchanged"), ("oracle:predicted", "oracle $-$ predicted")]
    metrics = [("rougeL", "ROUGE-L"), ("precision", "entity precision"), ("recall", "entity recall"),
               ("new_recall", "new-entity recall")]
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.4), sharey=True)
    for ax, (metric, name) in zip(axes, metrics):
        chosen = [r for r in rows if r["metric"] == metric and r["pair"] in dict(pairs)]
        span = max(max(abs(r["ci_low"]), abs(r["ci_high"])) for r in chosen) * 1.15
        ax.axvspan(0, span, color=BAND, zorder=0)
        for k, (pair, _) in enumerate(pairs):
            r = next(r for r in chosen if r["pair"] == pair)
            significant = r["p_holm"] < 0.05
            ax.plot([r["ci_low"], r["ci_high"]], [3 - k, 3 - k], color=stroke("blue"), lw=1.1)
            ax.plot(r["diff"], 3 - k, "o", ms=5.5, mec=stroke("blue"), mew=0.9,
                    color=stroke("blue") if significant else "white")
        ax.axvline(0, color=MUTED, lw=0.7, ls="--")
        ax.set_xlim(-span, span)
        ax.xaxis.set_major_formatter(SIGNED)
        ax.text(span * 0.95, -0.85, "A better", ha="right", fontsize=6.2, color=MUTED)
        ax.text(-span * 0.95, -0.85, "B better", ha="left", fontsize=6.2, color=MUTED)
        ax.grid(axis="y", visible=False)
        title(ax, name)
    axes[0].set_yticks(range(4), [label for _, label in pairs][::-1])
    axes[0].set_ylim(-1.1, 3.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=stroke("blue"), mec=stroke("blue"), ms=5.5),
               plt.Line2D([], [], marker="o", ls="", color="white", mec=stroke("blue"), ms=5.5)]
    fig.supxlabel("difference A $-$ B with 95% bootstrap interval over 50 articles", fontsize=8)
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.88))
    fig.legend(handles, ["Holm-corrected $p < 0.05$", "not significant after correction"], loc="upper center",
               ncol=2, bbox_to_anchor=(0.55, 1.0))
    fig.savefig(out / "differences.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", nargs="*", help="figure names to build")
    args = parser.parse_args()
    outputs, out, datasets = Path(args.outputs), Path(args.out), Path(args.datasets)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {
        "data_build": lambda: data_build(datasets, out),
        "aligner": lambda: aligner(datasets, out),
        "label_noise": lambda: label_noise(datasets, out),
        "classifier": lambda: classifier(outputs, out),
        "seeds": lambda: seeds(outputs, out),
        "stale_scores": lambda: stale_scores(outputs / "plots", out),
        "stale_rules": lambda: stale_rules(outputs / "plots", out),
        "stale_tokens": lambda: stale_tokens(outputs / "plots", out),
        "update_rouge": lambda: update_rouge(outputs, out),
        "differences": lambda: differences(outputs, out),
    }
    for name, job in jobs.items():
        if not args.only or name in args.only:
            job()
    if not args.only or "entities" in args.only:
        runs = {"oracle": outputs / "runs/fix-oracle", "noev": outputs / "runs/fix-oracle-noev",
                "predicted": outputs / "runs/fix-predicted"}
        ids, n_sentences, sums = collect(args.data, runs)
        entities(sums, len(ids), n_sentences["all"], out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
