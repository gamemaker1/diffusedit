"""Article-level metrics of Section 9.2 for any system's output, with paired tests.

A system is a directory with results.jsonl, one {"id", "output"} line per
article, where "output" is the list of output sentences. infilling.run writes
this format, and a baseline only has to write the same two fields. "copy",
the unchanged source article, is always included.

For one article with source A, gold article A*, evidence E and output Â,
U(·) keeps the sentences with no verbatim match in A after whitespace
normalization (infilling.faithfulness.updated).

    rouge1, rouge2, rougeL
                UpdateROUGE, the ROUGE F1 between the concatenations of U(Â)
                and U(A*). Articles with an empty U(A*) are skipped, and an empty
                U(Â) scores 0.
    entity_precision, entity_recall
                O and G are the mention sets of U(Â) and U(A*), with the
                extractor M of infilling.entities run on each sentence.
                precision |O & G| / |O|, recall |O & G| / |G|.
    unsupported
                The count of mentions in O that are neither mentions of A or
                E (M run on each source sentence and each snippet) nor
                whole-word strings of their text. Lower is better. The mention
                test covers digit strings inside words, such as the "3" of
                "3rd", and the string test covers mentions that M cuts
                differently in the two texts.
    faithfulness
                AlignScore of U(Â) against E, read from <system>/alignscore.jsonl,
                which infilling.faithfulness writes in its own environment.
                Articles with an empty U(Â) are skipped, so copy has none.
    edit_distance
                Word-level Levenshtein distance between Â and A, divided by
                the longer length. Words are \\w+ runs and single punctuation
                marks, case kept, so every system is cut the same way
                whatever its own tokenizer. Lower is closer to the source.
    keep_verbatim
                The fraction of gold-KEEP source sentences that occur
                verbatim, after whitespace normalization, among the output
                sentences.

With --cascades, the hand-labelled cascade file of Section 8.5 (the format of
cascading.evaluate), two more metrics use the sentences labelled "cascade".

    cascade_rouge1, cascade_rouge2, cascade_rougeL
                UpdateROUGE with U(Â) and U(A*) restricted to the cascade
                sentences, which compares output[j] with the gold target of j.
                This needs an output aligned to the source sentences, so a
                system whose output length differs from the source is skipped.
    cascade_recall_shared, cascade_recall_unshared
                The fraction of cascade sentences flagged in any round,
                read from the "flagged" lists of the result's "rounds" field.
                A result without that field counts as no flags. Split by
                shares_mention_with_direct.

Every metric is a ratio of sums over articles, so a mean is a sum of per-
article numerators over a sum of denominators. The paired tests of
infilling.significance compare systems with the article as the unit.

The token-level detection metrics of Section 9.2 come from classifier.train
and staleness.train with --eval.

Usage:
    python -m infilling.metrics ../datasets/synthetic/split/train.jsonl \\
        oracle=runs/fix-oracle noev=runs/fix-oracle-noev \\
        --pairs oracle:copy oracle:noev --out runs/metrics.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer

from classifier.inputs import read_jsonl
from infilling.entities import Extractor
from infilling.faithfulness import norm, updated
from infilling.significance import holm, test

ROUGE = ("rouge1", "rouge2", "rougeL")
METRICS = (*ROUGE, "entity_precision", "entity_recall", "unsupported", "faithfulness",
           "edit_distance", "keep_verbatim")
CASCADE_METRICS = (*(f"cascade_{m}" for m in ROUGE), "cascade_recall_shared", "cascade_recall_unshared")
LOWER_IS_BETTER = {"unsupported", "edit_distance"}


def words(text):
    return re.findall(r"\w+|[^\w\s]", text)


def levenshtein(a, b):
    """Minimum count of single-word insertions, deletions and substitutions turning a into b."""
    start = 0
    while start < min(len(a), len(b)) and a[start] == b[start]:
        start += 1
    end = 0
    while end < min(len(a), len(b)) - start and a[-1 - end] == b[-1 - end]:
        end += 1
    a, b = a[start:len(a) - end], b[start:len(b) - end]
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def occurs(mention, text):
    return re.search(rf"(?<!\w){re.escape(mention)}(?!\w)", text) is not None


def mentions(m, sentences):
    return set().union(*(m(s) for s in sentences)) if sentences else set()


def rouge(scorer, output, gold):
    """(F1 per ROUGE type, 1), or (0, 0) when the gold side is empty."""
    if not gold:
        return {k: (0.0, 0) for k in ROUGE}
    result = scorer.score(" ".join(gold), " ".join(output)) if output else None
    return {k: (result[k].fmeasure if result else 0.0, 1) for k in ROUGE}


def gold_rewrite(record, j):
    """The gold rewrite of source sentence j, or None for a deleted sentence."""
    entry = record["labels"][j]
    return entry["target"] if entry["label"] != "DROP" else None


def article_metrics(m, scorer, record, output, faithfulness, cascades, flagged):
    """Numerator and denominator of every metric for one system on one article."""
    source = [s.strip() for s in record["source_sentences"]]
    in_source = {norm(s) for s in source}
    targets = record["target_sentences"]
    out_u, gold_u = updated(output, in_source), updated(targets, in_source)
    values = rouge(scorer, out_u, gold_u)

    o, g = mentions(m, out_u), mentions(m, gold_u)
    values["entity_precision"] = (len(o & g), len(o))
    values["entity_recall"] = (len(o & g), len(g))
    snippets = [f"{s['title']}: {s['text']}" for s in record["evidence"]]
    known = mentions(m, source) | mentions(m, snippets)
    text = "\n".join(source + snippets).lower()
    values["unsupported"] = (sum(x not in known and not occurs(x, text) for x in o), 1)
    values["faithfulness"] = (faithfulness, 1) if faithfulness is not None and out_u else (0.0, 0)

    a, b = words(" ".join(source)), words(" ".join(s.strip() for s in output if s.strip()))
    values["edit_distance"] = (levenshtein(a, b) / max(len(a), len(b), 1), 1)
    keep = [s for s, lab in zip(source, record["labels"]) if lab["label"] == "KEEP"]
    in_output = {norm(s) for s in output}
    values["keep_verbatim"] = (sum(norm(s) in in_output for s in keep), len(keep))

    if cascades is not None:
        idx = [c["sentence_idx"] for c in cascades]
        if len(output) == len(source):
            gold = [g for g in (gold_rewrite(record, j) for j in idx) if g is not None]
            cascade = rouge(scorer, updated([output[j] for j in idx], in_source), updated(gold, in_source))
            values.update({f"cascade_{k}": v for k, v in cascade.items()})
        for group, shared in (("shared", True), ("unshared", False)):
            members = [c["sentence_idx"] for c in cascades if c["shares_mention_with_direct"] == shared]
            values[f"cascade_recall_{group}"] = (sum(j in flagged for j in members), len(members))
    return values


def load_cascades(path):
    """Cascade entries keyed by article id, keeping sentences hand-labelled "cascade"."""
    out = {}
    for row in read_jsonl(path):
        out[row["article_id"]] = [c for c in row["cascade_sentences"] if c["label"] == "cascade"]
    return out


def collect(data, run_dirs, m, cascade_path=None):
    """Article ids and sums[system][metric], an array [2, n_articles] of numerators and denominators."""
    runs = {name: {r["id"]: r for r in read_jsonl(Path(path) / "results.jsonl")} for name, path in run_dirs.items()}
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    records = {r["id"]: r for r in read_jsonl(data) if r["id"] in set(ids)}
    faith = {}
    for name, path in run_dirs.items():
        scores = Path(path) / "alignscore.jsonl"
        faith[name] = {r["id"]: r["score"] for r in read_jsonl(scores)} if scores.exists() else {}
    cascades = load_cascades(cascade_path) if cascade_path else None
    names = METRICS + (CASCADE_METRICS if cascades is not None else ())
    scorer = rouge_scorer.RougeScorer(list(ROUGE), use_stemmer=True)

    sums = {s: {k: np.zeros((2, len(ids))) for k in names} for s in ("copy", *runs)}
    for a, article_id in enumerate(ids):
        record = records[article_id]
        cascade = None if cascades is None else cascades.get(article_id, [])
        outputs = {"copy": (record["source_sentences"], None, set())}
        for name, run in runs.items():
            result = run[article_id]
            flagged = {j for r in result.get("rounds", []) for j in r.get("flagged", [])}
            outputs[name] = (result["output"], faith[name].get(article_id), flagged)
        for system, (output, faithfulness, flagged) in outputs.items():
            values = article_metrics(m, scorer, record, output, faithfulness, cascade, flagged)
            for k, (num, den) in values.items():
                sums[system][k][:, a] = (num, den)
    return ids, sums


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="the dataset split the runs read")
    parser.add_argument("runs", nargs="+", help="name=run_dir")
    parser.add_argument("--cascades", help="hand-labelled cascade JSONL of Section 8.5")
    parser.add_argument("--pairs", nargs="*", default=[], help="A:B, where copy is the unchanged article")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="JSON output path")
    args = parser.parse_args()

    ids, sums = collect(args.data, dict(spec.split("=", 1) for spec in args.runs), Extractor(), args.cascades)
    names = list(next(iter(sums.values())))

    def mean(num, den):
        return float(num.sum() / den.sum()) if den.sum() else None

    report = {"articles": len(ids), "resamples": args.resamples,
              "counted": {s: {k: int((v[k][1] > 0).sum()) for k in names} for s, v in sums.items()},
              "means": {s: {k: mean(*v[k]) for k in names} for s, v in sums.items()},
              "per_article": {s: {k: v[k].tolist() for k in names} for s, v in sums.items()}, "ids": ids}
    print(f"{len(ids)} articles")
    print(f"{'metric':24s}" + "".join(f"{s:>12s}" for s in sums))
    for k in names:
        cells = [report["means"][s][k] for s in sums]
        print(f"{k:24s}" + "".join(f"{v:12.4f}" if v is not None else f"{'-':>12s}" for v in cells))

    rng = np.random.default_rng(args.seed)
    rows = []
    for pair in args.pairs:
        a_name, b_name = pair.split(":")
        for k in names:
            (na, da), (nb, db) = sums[a_name][k], sums[b_name][k]
            if not da.sum() or not db.sum():
                continue
            keep = (da > 0) | (db > 0)
            diff, low, high, p = test(na[keep], da[keep], nb[keep], db[keep], rng, args.resamples)
            rows.append({"pair": pair, "metric": k, "articles": int(keep.sum()),
                         "diff": diff, "ci_low": low, "ci_high": high, "p": p})
    for row, adjusted in zip(rows, holm(np.array([r["p"] for r in rows]))):
        row["p_holm"] = float(adjusted)
    if rows:
        print(f"\n{'pair':22s} {'metric':24s} {'A-B':>8s} {'95% CI':>19s} {'p':>7s} {'p_holm':>7s}")
    for r in rows:
        mark = "*" if r["p_holm"] < 0.05 else " "
        print(f"{r['pair']:22s} {r['metric']:24s} {r['diff']:+8.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
              f"{r['p']:7.4f} {r['p_holm']:7.4f}{mark}")
    report["tests"] = rows
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
