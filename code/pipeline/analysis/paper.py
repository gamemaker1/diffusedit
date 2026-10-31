"""The figures of the final write-up, one finding per figure, from the pulled results.

    ladder.pdf    gain in sentence ROUGE-L over copying, per repair setting
    values.pdf    exact match, new values written and old values kept, per system
    cascade.pdf   cascade score and cascade sentences flagged per detector, and the loop by round
    focus.pdf     where the attention weight of the staleness head falls
    evidence.pdf  each head with and without evidence

Every bar starts at zero, in the pastel colours of the write-up. Grey marks a
setting that learns nothing, blue the DiffusEdit heads, violet an oracle, and
orange the fine-tuned editor. A thin
black line on a bar is the 95% bootstrap interval over articles, and a band
around a line is the same interval. The head scores carry the standard
deviation over five training seeds.

Usage:
    python -m analysis.paper --out ../../docs/final/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.extra import ratio, score
from classifier.inputs import read_jsonl

BLUE, ORANGE, AQUA, VIOLET, GRAY = "#8fb8e8", "#f4a582", "#8fd3c1", "#b39ddb", "#cfd4d9"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": False, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 9, "legend.fontsize": 8.5, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK, "axes.edgecolor": "#c8ccd0",
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "grid.color": "#e4e7ea", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "ytick.major.size": 0,
})
COL = 3.3
RESAMPLES = 2000


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def interval(num, den, seed=0):
    """Ratio of sums over articles and its 95% bootstrap interval."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(seed).integers(0, len(num), (RESAMPLES, len(num)))
    draws = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def gain(rows, seed=0):
    """Mean of chosen minus copy over sentences, with the article as the bootstrap unit."""
    ids = list(rows)
    num = np.array([sum(r["chosen"] - r["copy"] for r in rows[i]) for i in ids])
    den = np.array([len(rows[i]) for i in ids])
    return interval(num, den, seed)


def bars(ax, labels, values, colors, fmt, xmax, errors=None, title=None, height=0.62):
    """Horizontal bars from zero, the value written after each bar, with an interval line if given."""
    y = np.arange(len(labels))[::-1]
    for k, (yi, v, c) in enumerate(zip(y, values, colors)):
        if v != v:
            continue
        ax.barh(yi, v, height=height, color=c, linewidth=0)
        end = v
        if errors is not None and errors[k] is not None:
            lo, hi = errors[k]
            ax.plot([lo, hi], [yi, yi], color=INK, lw=1.0, solid_capstyle="butt")
            ax.plot([lo, lo], [yi - 0.12, yi + 0.12], color=INK, lw=1.0)
            ax.plot([hi, hi], [yi - 0.12, yi + 0.12], color=INK, lw=1.0)
            end = max(v, hi)
        ax.text(end + 0.012 * xmax, yi, fmt.format(v), va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlim(0, xmax)
    ax.grid(axis="x")
    if title:
        ax.set_title(title, loc="left", pad=8)


def pct_axis(ax, ticks=(0, 0.5, 1.0)):
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{t:.0%}" for t in ticks])


def ladder(records, runs, editor_rows, out):
    copy = json.load(open(runs / "fix-rerank/metrics.json"))["copy_rougeL"]
    rows = [("Linear masks, no infilling rules", "fix-predicted", GRAY), ("Attention masks, no infilling rules", "fix-attention", BLUE),
            ("Infilling rules, heuristic ranking", "fix-tuned", BLUE), ("Infilling rules, learned ranking", "fix-rerank", BLUE),
            ("Oracle masks, infilling rules, learned ranking (300 articles)", "fix-oracle-rerank", VIOLET), ("Flan-T5 editor, fine-tuned", None, ORANGE)]
    stats = []
    for _, run, _ in rows:
        if run is None:
            stats.append(gain(editor_rows))
        else:
            results = [r for r in read_jsonl(runs / run / "results.jsonl") if r["id"] in records]
            stats.append(gain(score(records, results, False)))
    fig, ax = plt.subplots(figsize=(COL, 1.1))
    bars(ax, [r[0] for r in rows], [s[0] for s in stats], [r[2] for r in rows], "+{:.3f}", 0.13,
         errors=[(s[1], s[2]) for s in stats], title=f"Gain in sentence ROUGE-L over copying ({copy:.3f})")
    ax.set_xticks([0, 0.025, 0.05, 0.075, 0.1])
    save(fig, out, "ladder")


def values(records, runs, editor_rows, out):
    systems = [("Copy", None, GRAY), ("Infilling rules, heuristic ranking", runs / "fix-tuned", BLUE),
               ("Infilling rules, learned ranking", runs / "fix-rerank", BLUE), ("Flan-T5 editor", None, ORANGE)]
    metrics = [("exact", "Output equals the gold rewrite"), ("new", "New-value recall"), ("old", "Old values kept (lower is better)")]
    table = {}
    for name, path, _ in systems:
        if name == "Copy":
            results = [{"id": i, "output": r["source_sentences"]} for i, r in records.items()]
        elif name == "Flan-T5 editor":
            table[name] = editor_rows
            continue
        else:
            results = [r for r in read_jsonl(path / "results.jsonl") if r["id"] in records]
        table[name] = score(records, results, False)
    fig, axes = plt.subplots(1, 3, figsize=(COL * 2.1, 1.2), sharey=True)
    for ax, (key, title) in zip(axes, metrics):
        stats = [ratio(table[n], key) for n, _, _ in systems]
        bars(ax, [n for n, _, _ in systems], [s[0] for s in stats], [c for _, _, c in systems], "{:.0%}", 1.25,
             errors=[(s[1], s[2]) for s in stats], title=title)
        pct_axis(ax)
    fig.subplots_adjust(wspace=0.12)
    save(fig, out, "values")


def cascade(numbers, loops, out):
    per = loops["per_article"]
    dets = [("Single round", "none", GRAY), ("Staleness delta", "staleness_delta", BLUE),
            ("Edited as evidence", "edited_as_evidence", BLUE), ("Consequence loop", "consequence", BLUE),
            ("Consequence loop, gold labels", "oracle_labels", VIOLET)]
    stats = [interval(*per[d[1]]["cascade_rougeL"]) for d in dets]
    recall = {k: numbers["cascades"][k]["recall"] for k in ("staleness_delta", "edited_as_evidence", "consequence")}
    recall["none"], recall["oracle_labels"] = [0.0, 0.0, 0.0], [float("nan")] * 3
    fig, axes = plt.subplots(1, 3, figsize=(COL * 2.1, 1.25), gridspec_kw={"width_ratios": [1.25, 0.9, 1.15]})
    bars(axes[0], [d[0] for d in dets], [s[0] for s in stats], [d[2] for d in dets], "{:.3f}", 0.5,
         errors=[(s[1], s[2]) for s in stats], title="Cascade ROUGE-L")
    axes[0].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    bars(axes[1], ["" for _ in dets], [recall[d[1]][0] for d in dets], [d[2] for d in dets], "{:.0%}", 1.25,
         errors=[(recall[d[1]][1], recall[d[1]][2]) for d in dets], title="Detector recall on cascades")
    pct_axis(axes[1])
    ax = axes[2]
    keys, x = ["round1", "round2", "round3"], [1, 2, 3]
    for metric, color, label, dy in (("keep_verbatim", AQUA, "KEEP verbatim", 0.11), ("cascade_rougeL", BLUE, "Cascade ROUGE-L", -0.15)):
        st = [interval(*per[k][metric]) for k in keys]
        ax.fill_between(x, [s[1] for s in st], [s[2] for s in st], color=color, alpha=0.18, linewidth=0)
        ax.plot(x, [s[0] for s in st], "-o", color=color, ms=6, lw=2)
        for xi, s in zip(x, st):
            ax.text(xi, s[0] + dy, f"{s[0]:.1%}" if metric == "keep_verbatim" else f"{s[0]:.3f}", ha="center", va="center", fontsize=8)
        ax.text(3.25, st[-1][0], label, ha="left", va="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["round 1", "round 2", "round 3"])
    ax.set_xlim(0.6, 4.6)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="y")
    ax.set_title("Consequence loop by round", loc="left", pad=8)
    fig.subplots_adjust(wspace=0.25)
    save(fig, out, "cascade")


def focus(f, out):
    g = f["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(COL * 2.1, 1.3), gridspec_kw={"width_ratios": [1.15, 1]})
    rows = [("Stale tokens", g["stale"]["value"], BLUE), ("    if the weight were uniform", g["stale"]["chance"], GRAY),
            ("Tokens that are not stale", g["clean"]["value"], BLUE), ("    if the weight were uniform", g["clean"]["chance"], GRAY)]
    bars(axes[0], [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], "{:.1%}", 0.55,
         title="Share of attention weight on the new value")
    pct_axis(axes[0], (0, 0.2, 0.4))
    forms = [("Stale, new value a year", g["stale_year"]["top_value"], BLUE), ("Stale, a number", g["stale_number"]["top_value"], BLUE),
             ("Stale, a name", g["stale_name"]["top_value"], BLUE), ("Not stale", g["clean"]["top_value"], GRAY)]
    bars(axes[1], [n for n, _, _ in forms], [v for _, v, _ in forms], [c for _, _, c in forms], "{:.0%}", 1.25,
         title="Most weighted token lies in the new value")
    pct_axis(axes[1])
    fig.subplots_adjust(wspace=0.75)
    save(fig, out, "focus")


def evidence(n, seeds, out):
    def sd(key):
        v = seeds.get(key)
        return float(np.std(v, ddof=1)) if v else None
    rows = [("Sentence classifier, macro-F1", n["classifier"]["ce24"]["macro_f1"], sd("ce24"),
             n["classifier"]["noev24"]["macro_f1"], sd("noev24")),
            ("Linear staleness classifier, PR-AUC", n["staleness"]["linear"]["pr_auc"], sd("stale-bce"),
             n["staleness"]["noev"]["pr_auc"], sd("noev-stale-bce")),
            ("Attention staleness classifier, PR-AUC", n["staleness"]["attention"]["pr_auc"], sd("attention"),
             n["staleness"]["controls"]["no_evidence"]["pr_auc"], None)]
    labels, vals, cols, errs = [], [], [], []
    for name, with_ev, with_sd, without, without_sd in rows:
        labels += [name, ""]
        vals += [with_ev, without]
        cols += [BLUE, GRAY]
        errs += [(with_ev - with_sd, with_ev + with_sd) if with_sd else None,
                 (without - without_sd, without + without_sd) if without_sd else None]
    fig, ax = plt.subplots(figsize=(COL, 1.4))
    bars(ax, labels, vals, cols, "{:.3f}", 1.05, errors=errs, height=0.75, title="Each classifier with and without the evidence")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.bar([0], [0], color=BLUE, label="with evidence")
    ax.bar([0], [0], color=GRAY, label="without evidence: retrained on the article alone, or, for the attention classifier, evidence slots emptied")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=7.5)
    save(fig, out, "evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", default="outputs/jl/full")
    parser.add_argument("--data", default="../dataset/split/test.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    outputs, out = Path(args.outputs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = outputs / "runs"
    numbers = json.load(open(outputs / "analysis/numbers.json"))
    seeds = dict(numbers["seeds"])
    attention_seeds = sorted((outputs / "heads/sweep/attention").glob("seed*/metrics.json"))
    if attention_seeds:
        seeds["attention"] = [json.load(open(f))["test"]["pr_auc"] for f in attention_seeds] + [numbers["staleness"]["attention"]["pr_auc"]]
    ids = set(json.load(open(outputs / "supported_ids.json")))
    records = {r["id"]: r for r in read_jsonl(args.data) if r["id"] in ids}
    editor = [r for r in read_jsonl(runs / "editor/results.jsonl") if r["id"] in records]
    editor_rows = score(records, editor, False)
    ladder(records, runs, editor_rows, out)
    values(records, runs, editor_rows, out)
    cascade(numbers, json.load(open(outputs / "analysis/loops.json")), out)
    focus(json.load(open(outputs / "analysis/focus.json")), out)
    evidence(numbers, seeds, out)
    print("figures in", out)


if __name__ == "__main__":
    main()
