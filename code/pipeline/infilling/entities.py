"""Entity metrics for an infilling run (Section 9.2).

M(text) is the mention extractor of datasets/collate.py: spaCy en_core_web_sm
entities of the types in MENTION_LABELS, lowercased and stripped, plus every
digit string. It runs on each sentence alone, so gold, source and output
mentions come from the same procedure.

For every repaired sentence, with G = M(gold rewrite), S = M(source sentence),
O = M(output) and E = the mentions of the article's evidence:

    precision        |O & G| / |O|
    recall           |O & G| / |G|
    new_recall       |O & (G - S)| / |G - S|, the entities the update adds
    stale_kept       |O & (S - G)| / |S - G|, outdated entities left in. Lower is better.
    unsupported      |O - S - E| / |O|, entities found in neither the source
                     nor the evidence. Lower is better.

Each is a micro-average over sentences (summed numerators over summed
denominators). The same numbers are reported for the unchanged source sentence
(copy) and for the candidate in the pool with the highest new_recall (pool).
Subsets: sentences with G - S non-empty, and the supported sentences of
infilling.evaluate.

Usage:
    python -m infilling.entities runs/fix-oracle ../datasets/collated/train.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from classifier.inputs import read_jsonl

MENTION_LABELS = {
    "CARDINAL", "DATE", "FAC", "GPE", "LOC", "MONEY", "NORP", "ORDINAL",
    "ORG", "PERCENT", "PERSON", "QUANTITY", "TIME",
}
MENTION_STRIP = " .,;:'\"()!?\t\n"
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


class Extractor:
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.cache = {}

    def __call__(self, text):
        if text not in self.cache:
            doc = self.nlp(text)
            out = {re.sub(r"\s+", " ", e.text.lower()).strip(MENTION_STRIP)
                   for e in doc.ents if e.label_ in MENTION_LABELS}
            out |= set(NUMBER_RE.findall(text))
            out.discard("")
            self.cache[text] = out
        return self.cache[text]


def words(text):
    return set(re.findall(r"\w+", text.lower()))


def counts(o, g, s, e):
    """Numerators and denominators of every metric for one output mention set."""
    new, old = g - s, s - g
    return Counter({
        "precision_num": len(o & g), "precision_den": len(o),
        "recall_num": len(o & g), "recall_den": len(g),
        "new_recall_num": len(o & new), "new_recall_den": len(new),
        "stale_kept_num": len(o & old), "stale_kept_den": len(old),
        "unsupported_num": len(o - s - e), "unsupported_den": len(o),
    })


def ratios(c):
    names = ("precision", "recall", "new_recall", "stale_kept", "unsupported")
    return {n: c[f"{n}_num"] / c[f"{n}_den"] if c[f"{n}_den"] else None for n in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("data")
    parser.add_argument("--limit", type=int, help="score only the first N articles of the run")
    parser.add_argument("--out", help="default <run>/entity_metrics.json")
    args = parser.parse_args()

    run = Path(args.run)
    results = list(read_jsonl(run / "results.jsonl"))[:args.limit]
    wanted = {r["id"] for r in results}
    records = {r["id"]: r for r in read_jsonl(args.data) if r["id"] in wanted}
    m = Extractor()

    groups = {name: {"output": Counter(), "copy": Counter(), "pool": Counter(), "sentences": 0}
              for name in ("all", "with_new_entities", "supported")}
    for result in results:
        record = records[result["id"]]
        evidence_text = [f"{s['title']}: {s['text']}" for s in record["evidence"]]
        e = set().union(*(m(t) for t in evidence_text)) if evidence_text else set()
        evidence_words = words(" ".join(f"{s['title']} {s['text']}" for s in record["evidence"]))
        for edit in result["edits"]:
            j = edit["sentence"]
            entry = record["labels"][j]
            target = record["target_sentences"][entry["target_idx"]]["text"]
            source, output = result["source"][j], result["output"][j]
            g, s, o = m(target), m(source), m(output)
            texts = [c["text"] for d in edit["decisions"] for c in d["candidates"]]
            best = max(texts, key=lambda t: (len(m(t) & (g - s)), -len(m(t) - g)))
            new_words = words(target) - words(source)
            supported = bool(new_words) and new_words <= evidence_words | words(source)
            for name, member in (("all", True), ("with_new_entities", bool(g - s)), ("supported", supported)):
                if member:
                    groups[name]["output"] += counts(o, g, s, e)
                    groups[name]["copy"] += counts(s, g, s, e)
                    groups[name]["pool"] += counts(m(best), g, s, e)
                    groups[name]["sentences"] += 1

    metrics = {"articles": len(results)}
    for name, group in groups.items():
        metrics[name] = {"sentences": group["sentences"],
                         **{k: ratios(group[k]) for k in ("output", "copy", "pool")}}
        print(f"{name}: {group['sentences']} sentences")
        for k in ("output", "copy", "pool"):
            line = "  ".join(f"{n} {v:.3f}" if v is not None else f"{n} -" for n, v in metrics[name][k].items())
            print(f"  {k:6s} {line}")
    out = Path(args.out) if args.out else run / "entity_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
