"""Figures and numbers for docs/internal/experiments/synthetic/synthetic.tex.

Every figure compares the FRUIT runs (outputs/jl) with the synthetic runs
(outputs/jl/synthetic) from saved outputs. Nothing here uses the GPU. The
style is that of analysis.progress_figures. FRUIT is blue and the synthetic
set is orange in every figure.

    data.pdf         labels, stale share and supported share of both train sets
    probe.pdf        shortcut probe on both train sets (needs --probe)
    classifier.pdf   per-class F1 of the sentence classifiers, and evidence inputs
    confusion.pdf    confusion matrices of the block-24 classifier
    seeds.pdf        five seeds of the classifier and the staleness head
    stale_scores.pdf staleness score distributions and share stale per score bin
    stale_rules.pdf  precision against recall, and precision and recall per threshold
    infilling.pdf    ROUGE-L, hit and coverage of the infilling runs
    entities.pdf     entity measures of the infilling runs
    differences.pdf  paired differences between infilling systems with 95% intervals
    summary.pdf      the main measures on both train sets

The numbers the report quotes go to <outputs>/synthetic/analysis.json.

Usage:
    python -m analysis.synthetic --fruit ../datasets/collated/train.jsonl \\
        --synthetic ../datasets/synthetic/train.jsonl --outputs outputs/jl \\
        --out ../../docs/internal/experiments/synthetic/figures
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from analysis.classifier_tests import decide
from analysis.progress_figures import (BAND, GRID, INK, LABELS, MUTED, SEEDS, SIGNED, fill, load, plt, stroke,
                                       title)
from infilling.evaluate import gold, words

SETS = (("fruit", "FRUIT", "blue"), ("synthetic", "Synthetic", "orange"))
SYSTEMS = (("copy", "Copy"), ("oracle", "Oracle\nmasks"), ("noev", "Oracle,\nno evidence"),
           ("predicted", "Predicted\nmasks"))
RUNS = {"oracle": "fix-oracle", "noev": "fix-oracle-noev", "predicted": "fix-predicted"}
ENTITY = (("precision", "Entity\nprecision"), ("recall", "Entity\nrecall"), ("new_recall", "New-entity\nrecall"),
          ("stale_kept", "Outdated\nkept"), ("unsupported", "Fabricated"))


def root(outputs, key):
    return outputs if key == "fruit" else outputs / "synthetic"


def legend(ax, **kw):
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=fill(c), edgecolor=stroke(c), linewidth=0.8) for _, _, c in SETS]
    ax.legend(handles, [name for _, name, _ in SETS], **kw)


def pair_bars(ax, values, labels, fmt="{:.3f}", width=0.38, fontsize=6.2):
    """Grouped bars, one group per label and one bar per train set, each with its value above."""
    x = np.arange(len(labels))
    for k, (key, _, color) in enumerate(SETS):
        pos = x + (k - 0.5) * (width + 0.03)
        ax.bar(pos, values[key], width, color=fill(color), edgecolor=stroke(color), linewidth=0.8)
        for p, v in zip(pos, values[key]):
            ax.text(p, v + 0.012 * ax.get_ylim()[1], fmt.format(v), ha="center", va="bottom", fontsize=fontsize)
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------- data


def data_stats(path, sample, seed=42):
    """Label counts, stale share per EDIT sentence and supported share of one train set."""
    with open(path) as f:
        lines = f.readlines()
    if sample and sample < len(lines):
        lines = random.Random(seed).sample(lines, sample)
    labels, shares, supported, with_new = Counter(), [], 0, 0
    sentences, snippets = [], []
    for line in lines:
        record = json.loads(line)
        sentences.append(len(record["source_sentences"]))
        snippets.append(len(record["evidence"]))
        evidence = words(" ".join(f"{s['title']} {s['text']}" for s in record["evidence"]))
        for j, (source, entry) in enumerate(zip(record["source_sentences"], record["labels"])):
            labels[entry["label"]] += 1
            if entry["label"] != "EDIT":
                continue
            shares.append(sum(b - a for a, b in entry["stale_spans"]) / max(len(source), 1))
            new = words(gold(record, j, "target")) - words(source)
            with_new += bool(new)
            supported += bool(new) and new <= evidence | words(source)
    total = sum(labels.values())
    return {"articles": len(lines), "sentences": total, "labels": {k: labels[k] for k in LABELS},
            "label_share": {k: labels[k] / total for k in LABELS},
            "stale_share_mean": float(np.mean(shares)), "stale_share_hist": np.histogram(shares, np.linspace(0, 1, 21))[0].tolist(),
            "edit_with_new_words": with_new, "supported": supported, "supported_share": supported / with_new,
            "sentences_per_article": float(np.mean(sentences)), "snippets_per_article": float(np.mean(snippets))}


def data(stats, out):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [1.5, 1.5, 0.9]})
    ax1.set_ylim(0, 0.78)
    pair_bars(ax1, {k: [stats[k]["label_share"][c] for c in LABELS] for k, _, _ in SETS}, LABELS, "{:.1%}")
    ax1.set_ylabel("share of sentences")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    legend(ax1, loc="upper right")
    title(ax1, "(a) Sentence labels")

    edges = np.linspace(0, 1, 21)
    for key, name, color in SETS:
        h = np.array(stats[key]["stale_share_hist"], dtype=float)
        ax2.stairs(h / h.sum(), edges, fill=True, color=fill(color), alpha=0.55)
        ax2.stairs(h / h.sum(), edges, color=stroke(color), linewidth=1.1)
        ax2.axvline(stats[key]["stale_share_mean"], color=stroke(color), linestyle=(0, (3, 2)), linewidth=0.9)
        ax2.text(stats[key]["stale_share_mean"] + 0.015, 0.31 if key == "fruit" else 0.36,
                 f"{name} mean {stats[key]['stale_share_mean']:.1%}", fontsize=6.2, color=INK)
    ax2.set_ylim(0, 0.4)
    ax2.set_xlabel("stale share of an EDIT sentence's characters")
    ax2.set_ylabel("share of EDIT sentences")
    ax2.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    title(ax2, "(b) Stale share per EDIT sentence")

    ax3.set_ylim(0, 1.12)
    for k, (key, name, color) in enumerate(SETS):
        v = stats[key]["supported_share"]
        ax3.bar(k, v, 0.6, color=fill(color), edgecolor=stroke(color), linewidth=0.8)
        ax3.text(k, v + 0.02, f"{v:.1%}", ha="center", fontsize=6.6)
    ax3.set_xticks([0, 1], [name for _, name, _ in SETS])
    ax3.set_ylabel("share of EDIT sentences with new words")
    ax3.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax3.grid(axis="x", visible=False)
    title(ax3, "(c) Supported")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "data.pdf")
    plt.close(fig)


def read_probe(path):
    """Macro-F1 per feature set from the output of datasets/synthetic/probe.py, FRUIT sample first."""
    blocks = re.split(r"^== ", Path(path).read_text(), flags=re.M)[1:]
    result = {}
    for (key, _, _), block in zip(SETS, blocks):
        result[key] = {name.strip(): {"macro": float(macro), "KEEP": float(k), "EDIT": float(e), "DROP": float(d)}
                       for name, macro, k, e, d in re.findall(
                           r"^\s+(.+?)\s+macro-F1 ([\d.]+)\s+KEEP ([\d.]+) EDIT ([\d.]+) DROP ([\d.]+)", block, flags=re.M)}
    return result


def probe(result, out):
    names = ("sentence words only", "evidence-match flags only", "both")
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    ax.set_ylim(0, 0.62)
    pair_bars(ax, {k: [result[k][n]["macro"] for n in names] for k, _, _ in SETS},
              ["Sentence words", "Evidence-match flags", "Both"])
    ax.axhline(1 / 3, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(2.48, 1 / 3 + 0.008, "uniform guess", ha="right", fontsize=6.2, color=MUTED)
    ax.set_ylabel("probe macro-F1")
    legend(ax, loc="upper left", ncols=2)
    title(ax, "Linear probe on surface features")
    fig.tight_layout()
    fig.savefig(out / "probe.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- classifier


def held(metrics):
    return metrics.get("heldout") or metrics.get("val")


def classifier(outputs, out):
    models = (("MiniLM baseline", "heads/baseline"), ("LLaDA block 24", "heads/classifier/layer24"),
              ("LLaDA block 32", "heads/classifier/layer32"))
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    numbers = {}
    for ax, (name, path), tag in zip(axes, models, "abc"):
        values = {}
        for key, _, _ in SETS:
            v = held(load(root(outputs, key) / path / "metrics.json"))
            values[key] = [v[c]["f1"] for c in LABELS] + [v["macro_f1"]]
            numbers.setdefault(key, {})[name] = {**{c: v[c]["f1"] for c in LABELS}, "macro": v["macro_f1"],
                                                 "drop_precision": v["DROP"]["precision"],
                                                 "support": [v[c]["support"] for c in LABELS]}
        ax.set_ylim(0, 1.1)
        pair_bars(ax, values, [*LABELS, "macro"], "{:.2f}", fontsize=5.0)
        ax.tick_params(axis="x", labelsize=6.6)
        ax.axvspan(2.5, 3.5, color=BAND, zorder=0)
        if ax is axes[0]:
            ax.set_ylabel("held-out F1")
            legend(ax, loc="upper left", fontsize=6.4)
        else:
            ax.tick_params(labelleft=False)
        title(ax, f"({tag}) {name}")

    ax = axes[3]
    inputs = (("None", "classifier"), ("Mean", "evidence"), ("Nearest", "evidence_nearest"), ("Attention", "evidence_attention"))
    numbers["evidence"] = {}
    for key, _, color in SETS:
        for layer, style, marker in ((24, "-", "o"), (32, (0, (3, 2)), "s")):
            v = [held(load(root(outputs, key) / f"heads/{d}/layer{layer}/metrics.json"))["macro_f1"] for _, d in inputs]
            numbers["evidence"][f"{key}{layer}"] = dict(zip([n for n, _ in inputs], v))
            ax.plot(range(4), v, color=stroke(color), linestyle=style, linewidth=1.3, marker=marker, markersize=3.6,
                    markerfacecolor=fill(color), markeredgewidth=0.8)
            if key == "synthetic":
                ax.text(3.12, v[-1], f"block {layer}", fontsize=5.8, va="center", color=INK)
        if key == "fruit":
            ax.text(3.12, v[-1] + 0.005, "blocks 24\nand 32", fontsize=5.8, va="center", color=INK)
    ax.set_xticks(range(4), ["None", "Mean", "Near.", "Attn."], fontsize=6.4)
    ax.set_xlim(-0.3, 4.3)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("held-out macro-F1")
    title(ax, "(d) Evidence input")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(out / "classifier.pdf")
    plt.close(fig)
    return numbers


def confusion(outputs, out):
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.5))
    numbers = {}
    for ax, (key, name, color), tag in zip(axes, SETS, "ab"):
        d = np.load(root(outputs, key) / "heads/sweep/cls-ce/layer24/seed42/heldout_predictions.npz")
        y, pred = d["labels"].astype(int), decide(d["probs"].copy(), float(d["tau_d"]))
        m = np.array([[np.sum((y == a) & (pred == b)) for b in range(3)] for a in range(3)], dtype=float)
        share = m / m.sum(1, keepdims=True)
        numbers[key] = {"counts": m.astype(int).tolist(), "row_share": share.tolist()}
        cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("ramp", ["#FFFFFF", stroke(color)])
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{share[a, b]:.0%}\n{int(m[a, b]):,}", ha="center", va="center", fontsize=6.4,
                        color="white" if share[a, b] > 0.6 else INK)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("predicted label")
        if ax is axes[0]:
            ax.set_ylabel("derived label")
        ax.grid(visible=False)
        for side in ax.spines.values():
            side.set_visible(False)
        title(ax, f"({tag}) {name}, block 24")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "confusion.pdf")
    plt.close(fig)
    return numbers


def seeds(outputs, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.9, 1, 1]})
    configs = (("MiniLM\nbaseline", "baseline", "macro"), ("block 24\ncross-entropy", "cls-ce/layer24", "macro"),
               ("block 24\nfocal", "cls-focal/layer24", "macro"), ("block 32\ncross-entropy", "cls-ce/layer32", "macro"),
               ("block 32\nfocal", "cls-focal/layer32", "macro"))
    stale = (("binary\ncross-entropy", "stale-bce"), ("focal", "stale-focal"))
    rng = np.random.default_rng(1)
    numbers = {}

    def dots(ax, k, v, color, offset):
        ax.scatter(k + offset + rng.uniform(-0.07, 0.07, len(v)), v, s=13, color=fill(color), edgecolor=stroke(color),
                   linewidth=0.7, zorder=3)
        ax.plot([k + offset - 0.15, k + offset + 0.15], [v.mean()] * 2, color=stroke(color), lw=1.5)
        ax.text(k + offset, v.max() + 0.02, f"{v.mean():.3f}", ha="center", fontsize=5.6)

    for j, (key, _, color) in enumerate(SETS):
        sweep = root(outputs, key) / "heads/sweep"
        for k, (name, path, _) in enumerate(configs):
            v = np.array([held(load(sweep / path / f"seed{s}/metrics.json"))["macro_f1"] for s in SEEDS])
            numbers[f"{key}/{path}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
            dots(axes[0], k, v, color, (j - 0.5) * 0.4)
        for ax, metric in ((axes[1], "val_auroc"), (axes[2], "val_pr_auc")):
            for k, (name, path) in enumerate(stale):
                v = np.array([load(sweep / path / f"seed{s}/metrics.json")[metric] for s in SEEDS])
                numbers[f"{key}/{path}/{metric}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
                dots(ax, k, v, color, (j - 0.5) * 0.4)
    axes[0].set_xticks(range(5), [c[0] for c in configs], fontsize=6.2)
    axes[0].set_ylabel("held-out macro-F1")
    axes[0].set_ylim(0.38, 0.95)
    legend(axes[0], loc="center left", fontsize=6.4, bbox_to_anchor=(0.0, 0.62))
    title(axes[0], "(a) Sentence classifier, 5 seeds each")
    for ax, name, tag, lim in ((axes[1], "AUROC", "b", (0.5, 1.07)), (axes[2], "PR-AUC", "c", (0.3, 1.07))):
        ax.set_xticks([0, 1], [s[0] for s in stale], fontsize=6.2)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(*lim)
        ax.set_ylabel(f"held-out {name}")
        title(ax, f"({tag}) Staleness {name}")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "seeds.pdf")
    plt.close(fig)
    return numbers


# ---------------------------------------------------------------- staleness


def auroc(y, p):
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks over ties
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, ranks)
    ranks = (sums / counts)[inverse]
    pos = y.sum()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


def stale_stats(path):
    """Score distributions and the threshold curve from saved held-out token scores.

    The saved rows index tokens, not sentences, so per-sentence masking rules need the features on the instance.
    """
    d = np.load(path)
    p, y, tau = d["probs"].astype(np.float64), d["labels"].astype(int), float(d["tau"])
    base = float(y.mean())
    edges = np.linspace(0, 1, 26)
    which = np.clip(np.digitize(p, edges) - 1, 0, 24)
    count = np.bincount(which, minlength=25)
    stale = np.bincount(which, y, minlength=25)
    curve = []
    def point(t):
        m = p >= t
        tp = int((m & (y == 1)).sum())
        return {"tau": float(t), "precision": tp / max(int(m.sum()), 1), "recall": tp / int(y.sum()),
                "masked": float(m.mean()), "count": int(m.sum())}

    for t in np.round(np.arange(0.02, 0.985, 0.02), 2):
        if point(t)["count"] < 100:  # too few masked tokens for a precision
            break
        m = p >= t
        tp = int((m & (y == 1)).sum())
        curve.append({"tau": float(t), "precision": tp / max(int(m.sum()), 1), "recall": tp / int(y.sum()),
                      "masked": float(m.mean())})
    match = min(curve, key=lambda c: abs(c["masked"] - base))

    return {"tokens": len(y), "base_rate": base, "tau": tau, "auroc": auroc(y, p), "edges": edges.tolist(),
            "hist_stale": np.bincount(which[y == 1], minlength=25).tolist(),
            "hist_correct": np.bincount(which[y == 0], minlength=25).tolist(),
            "bin_count": count.tolist(), "bin_stale_share": (stale / np.maximum(count, 1)).tolist(),
            "curve": curve, "tuned": point(tau), "tau_match": match}


def stale_scores(stats, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
    for ax, (key, name, color), tag in zip(axes, SETS, "ab"):
        s = stats[key]
        edges = np.array(s["edges"])
        for hist, hue, label in ((s["hist_correct"], "gray", "correct"), (s["hist_stale"], "red", "stale")):
            h = np.array(hist, dtype=float)
            ax.stairs(h / h.sum(), edges, fill=True, color=fill(hue), alpha=0.6)
            ax.stairs(h / h.sum(), edges, color=stroke(hue), linewidth=1.1, label=f"{label} tokens")
        ax.axvline(s["tau"], color=INK, linestyle=(0, (3, 2)), linewidth=0.8)
        ax.text(s["tau"] - 0.02, 0.93, f"tuned $\\tau$ = {s['tau']:.2f}", transform=ax.get_xaxis_transform(),
                ha="right", fontsize=6.2)
        ax.set_xlabel("staleness score")
        ax.set_xlim(0, 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        if ax is axes[0]:
            ax.set_ylabel("share of the class's tokens")
        else:
            ax.legend(loc="center", bbox_to_anchor=(0.45, 0.55))
        title(ax, f"({tag}) {name}, AUROC {s['auroc']:.3f}")
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(0.56, 0.47, "calibrated", fontsize=6.2, color=MUTED, rotation=38)
    for key, name, color in SETS:
        s = stats[key]
        centre = (np.array(s["edges"])[:-1] + np.array(s["edges"])[1:]) / 2
        keep = np.array(s["bin_count"]) >= 50
        ax.plot(centre[keep], np.array(s["bin_stale_share"])[keep], color=stroke(color), linewidth=1.3, marker="o",
                markersize=3, markerfacecolor=fill(color), markeredgewidth=0.7, label=name)
    ax.set_xlabel("staleness score")
    ax.set_ylabel("share of tokens that are stale")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left")
    title(ax, "(c) Share stale per score bin")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "stale_scores.pdf")
    plt.close(fig)


def stale_rules(stats, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
    ax = axes[0]
    for key, name, color in SETS:
        s = stats[key]
        ax.plot([c["recall"] for c in s["curve"]], [c["precision"] for c in s["curve"]], color=stroke(color),
                linewidth=1.4, label=name)
        ax.axhline(s["base_rate"], color=stroke(color), linestyle=(0, (1, 2)), linewidth=0.8)
        ax.scatter([s["tuned"]["recall"]], [s["tuned"]["precision"]], s=22, color=fill(color), edgecolor=stroke(color),
                   linewidth=0.9, zorder=3)
        ax.text(0.02, s["base_rate"] + 0.015, f"{name} stale share {s['base_rate']:.1%}", fontsize=5.8, color=MUTED)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.42))
    title(ax, "(a) Precision against recall")
    for ax, (key, name, color), tag in zip(axes[1:], SETS, "bc"):
        s = stats[key]
        taus = [c["tau"] for c in s["curve"]]
        for metric, style, label in (("precision", "-", "precision"), ("recall", (0, (4, 2)), "recall"),
                                     ("masked", (0, (1, 1.5)), "share masked")):
            ax.plot(taus, [c[metric] for c in s["curve"]], color=stroke(color), linestyle=style, linewidth=1.3, label=label)
        ax.axhline(s["base_rate"], color=MUTED, linewidth=0.7)
        ax.axvline(s["tau"], color=INK, linestyle=(0, (3, 2)), linewidth=0.8)
        ax.set_xlabel("threshold $\\tau$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="center right" if key == "synthetic" else "upper right", fontsize=6.4,
                  bbox_to_anchor=(1.0, 0.52) if key == "synthetic" else None)
        title(ax, f"({tag}) {name}")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "stale_rules.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- infilling


def infilling_numbers(outputs):
    numbers = {}
    for key, _, _ in SETS:
        runs = root(outputs, key) / "runs"
        n = {}
        for system, run in RUNS.items():
            m = load(runs / run / "metrics.json")
            e = load(runs / run / "entity_metrics.json")["all"]
            n[system] = {"rougeL": m["chosen_rougeL"], "pool_rougeL": m["pool_rougeL"], "hit": m["hit"],
                         "coverage": m["chosen_coverage"], "pool_coverage": m["pool_coverage"],
                         "masked": m["masked_fraction"], "stale": m["stale_fraction"],
                         "seconds": m["seconds_per_article"], "sentences": m["sentences"], **e["output"],
                         "pool": e["pool"]}
            n["copy"] = {"rougeL": m["copy_rougeL"], **e["copy"]}
        n["tests"] = load(runs / "significance.json")
        numbers[key] = n
    return numbers


def infilling(numbers, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.5, 1.1, 1.1]})
    ax = axes[0]
    ax.set_ylim(0.5, 1.02)
    pair_bars(ax, {k: [numbers[k][s]["rougeL"] for s, _ in SYSTEMS] for k, _, _ in SETS}, [n for _, n in SYSTEMS])
    for bar_ in ax.patches:
        bar_.set_y(0.5)
        bar_.set_height(bar_.get_height() - 0.5)
    for t in ax.texts:
        t.set_y(t.get_position()[1] - 0.012 * 1.02 + 0.006)
    ax.tick_params(axis="x", labelsize=6.2)
    ax.set_ylabel("ROUGE-L against the gold rewrite")
    legend(ax, loc="upper left", ncols=2, fontsize=6.4)
    title(ax, "(a) ROUGE-L, axis from 0.5")
    for ax, metric, name, tag in ((axes[1], "hit", "hit", "b"), (axes[2], "coverage", "coverage of the chosen candidate", "c")):
        systems = SYSTEMS[1:]
        ax.set_ylim(0, 1.0)
        pair_bars(ax, {k: [numbers[k][s][metric] for s, _ in systems] for k, _, _ in SETS}, [n for _, n in systems],
                  "{:.0%}")
        ax.tick_params(axis="x", labelsize=6.2)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_ylabel(name)
        title(ax, f"({tag}) {name.split(' ')[0].capitalize()}")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "infilling.pdf")
    plt.close(fig)


def entities(numbers, out):
    fig, axes = plt.subplots(1, 5, figsize=(6.8, 2.3), sharey=True)
    for ax, (metric, name), tag in zip(axes, ENTITY, "abcde"):
        ax.set_ylim(0, 1.12)
        pair_bars(ax, {k: [numbers[k][s][metric] for s, _ in SYSTEMS] for k, _, _ in SETS},
                  ["copy", "oracle", "no ev.", "pred."], "{:.2f}", width=0.36, fontsize=4.8)
        ax.tick_params(axis="x", labelsize=5.6)
        lower = metric in ("stale_kept", "unsupported")
        title(ax, f"({tag}) {name.replace(chr(10), ' ')}")
        ax.title.set_fontsize(7)
        ax.text(0.98, 0.97, "lower is better" if lower else "higher is better", transform=ax.transAxes, ha="right",
                va="top", fontsize=5.6, color=MUTED)
    axes[0].set_ylabel("micro-average over sentences")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=fill(c), edgecolor=stroke(c), linewidth=0.8) for _, _, c in SETS]
    fig.legend(handles, [n for _, n, _ in SETS], loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(w_pad=0.4)
    fig.savefig(out / "entities.pdf")
    plt.close(fig)


def differences(numbers, out):
    pairs = (("oracle:copy", "Oracle $-$ copy"), ("oracle:noev", "Oracle $-$ no evidence"),
             ("predicted:copy", "Predicted $-$ copy"), ("oracle:predicted", "Oracle $-$ predicted"))
    metrics = (("rougeL", "ROUGE-L"), ("recall", "Entity recall"), ("new_recall", "New-entity recall"),
               ("unsupported", "Fabricated"))
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.5), sharey=True)
    for ax, (metric, name), tag in zip(axes, metrics, "abcd"):
        for j, (key, _, color) in enumerate(SETS):
            rows = {r["pair"]: r for r in numbers[key]["tests"]["tables"]["all"] if r["metric"] == metric}
            for k, (pair, _) in enumerate(pairs):
                r, y = rows[pair], len(pairs) - 1 - k + (0.5 - j) * 0.32
                ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=stroke(color), linewidth=1.3)
                ax.scatter([r["diff"]], [y], s=20, zorder=3, linewidth=0.9, edgecolor=stroke(color),
                           color=stroke(color) if r["p_holm"] < 0.05 else "white")
        ax.axvline(0, color=INK, linewidth=0.7)
        ax.xaxis.set_major_formatter(SIGNED)
        ax.grid(axis="y", visible=False)
        title(ax, f"({tag}) {name}")
    axes[0].set_yticks(range(len(pairs)), [n for _, n in reversed(pairs)], fontsize=6.4)
    handles = [plt.Line2D([], [], color=stroke(c), marker="o", markersize=4, linewidth=1.3) for _, _, c in SETS]
    handles += [plt.Line2D([], [], color=MUTED, marker="o", markersize=4, linewidth=0, markerfacecolor=f, markeredgecolor=MUTED)
                for f in (MUTED, "white")]
    fig.legend(handles, [*(n for _, n, _ in SETS), "$p_{Holm} < 0.05$", "not significant"], loc="lower center",
               ncols=4, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=0.8)
    fig.savefig(out / "differences.pdf")
    plt.close(fig)


def summary(cls, stale, infill, out):
    rows = (("MiniLM baseline, macro-F1", lambda k: cls[k]["MiniLM baseline"]["macro"]),
            ("LLaDA block 32, macro-F1", lambda k: cls[k]["LLaDA block 32"]["macro"]),
            ("LLaDA block 24, macro-F1", lambda k: cls[k]["LLaDA block 24"]["macro"]),
            ("Staleness AUROC", lambda k: stale[k]["auroc"]),
            ("Staleness precision at tuned $\\tau$", lambda k: stale[k]["tuned"]["precision"]),
            ("ROUGE-L, copy", lambda k: infill[k]["copy"]["rougeL"]),
            ("ROUGE-L, oracle masks", lambda k: infill[k]["oracle"]["rougeL"]),
            ("ROUGE-L, predicted masks", lambda k: infill[k]["predicted"]["rougeL"]),
            ("New-entity recall, oracle masks", lambda k: infill[k]["oracle"]["new_recall"]),
            ("New-entity recall, predicted masks", lambda k: infill[k]["predicted"]["new_recall"]),
            ("Hit, oracle masks", lambda k: infill[k]["oracle"]["hit"]))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i, (name, get) in enumerate(rows):
        y = len(rows) - 1 - i
        a, b = get("fruit"), get("synthetic")
        ax.plot([a, b], [y, y], color=GRID, linewidth=2.4, zorder=1, solid_capstyle="round")
        for (key, _, color), v in zip(SETS, (a, b)):
            ax.scatter([v], [y], s=34, color=fill(color), edgecolor=stroke(color), linewidth=1.0, zorder=3)
        ax.text(min(a, b) - 0.02, y, f"{a:.3f}" if a < b else f"{b:.3f}", ha="right", va="center", fontsize=6.4)
        ax.text(max(a, b) + 0.02, y, f"{b:.3f}" if a < b else f"{a:.3f}", ha="left", va="center", fontsize=6.4)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlim(0, 1.1)
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", linewidth=0, markersize=5.5, markerfacecolor=fill(c), markeredgecolor=stroke(c))
               for _, _, c in SETS]
    ax.legend(handles, [n for _, n, _ in SETS], loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    title(ax, "Held-out scores on both train sets")
    fig.tight_layout()
    fig.savefig(out / "summary.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fruit", required=True, help="collated FRUIT train.jsonl")
    parser.add_argument("--synthetic", required=True, help="synthetic train.jsonl")
    parser.add_argument("--outputs", default="outputs/jl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--probe", help="output of datasets/synthetic/probe.py on a FRUIT sample, then the synthetic set")
    args = parser.parse_args()
    outputs, out = Path(args.outputs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    numbers = {"data": {"synthetic": data_stats(args.synthetic, None)}}
    numbers["data"]["fruit"] = data_stats(args.fruit, numbers["data"]["synthetic"]["articles"])
    data(numbers["data"], out)
    if args.probe:
        numbers["probe"] = read_probe(args.probe)
        probe(numbers["probe"], out)
    numbers["classifier"] = classifier(outputs, out)
    numbers["confusion"] = confusion(outputs, out)
    numbers["seeds"] = seeds(outputs, out)
    numbers["classifier_tests"] = {k: load(root(outputs, k) / "heads/sweep/tests.json") for k, _, _ in SETS}
    numbers["staleness"] = {k: stale_stats(root(outputs, k) / "heads/sweep/stale-bce/seed42/heldout_predictions.npz")
                            for k, _, _ in SETS}
    stale_scores(numbers["staleness"], out)
    stale_rules(numbers["staleness"], out)
    numbers["infilling"] = infilling_numbers(outputs)
    infilling(numbers["infilling"], out)
    entities(numbers["infilling"], out)
    differences(numbers["infilling"], out)
    summary(numbers["classifier"], numbers["staleness"], numbers["infilling"], out)
    (outputs / "synthetic/analysis.json").write_text(json.dumps(numbers, indent=1))
    print(f"figures in {out}, numbers in {outputs / 'synthetic/analysis.json'}")


if __name__ == "__main__":
    main()
