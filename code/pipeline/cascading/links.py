"""Cascade labels and consequence-head pairs from synthetic cascade links.

A synthetic record lists its injected pairs in "cascades", one
{"direct": j, "cascade": k} link each, so the cascades of Section 8.5 are
known without hand labels or mining. The direct edits of such an article are
its EDIT and DROP sentences that are not the cascade of a link. A record
without links has no cascades, and all its EDIT and DROP sentences are
direct edits.

A seeded fraction of the articles with a link becomes the evaluation set and
the rest give training pairs, so no article is in both.

    labels.jsonl  the format of cascading.evaluate, one line per evaluation
                  article, with its direct edits and its cascade sentences.
                  shares_mention_with_direct follows datasets/collate.py.
                  It is true when the mentions of both versions of k meet
                  the mentions of both versions of some direct edit.
    pairs.jsonl   one ordered pair per line, {"article_id", "j", "k", "label"}.
                  Each link (j, k) is a positive, and each positive draws up
                  to NEG_PER_POS KEEP sentences k' of its article as negative
                  pairs (j, k'), as in Section 6.4.

Usage:
    python -m cascading.links ../datasets/synthetic/cascade/source.jsonl ../datasets/synthetic/cascade
"""

import argparse
import json
import random
from pathlib import Path

from cascading.mentions import get_extractor
from classifier.inputs import read_jsonl

NEG_PER_POS = 3


def target(record, i):
    """The gold rewrite of source sentence i, or "" for a deleted sentence."""
    entry = record["labels"][i]
    return entry["target"] if entry["label"] != "DROP" else ""


def direct(record):
    """Indices of the direct edits, the changed sentences whose change the evidence supports."""
    cascades = {link["cascade"] for link in record.get("cascades", [])}
    return sorted(i for i, entry in enumerate(record["labels"])
                  if entry["label"] != "KEEP" and i not in cascades)


def label_row(record, m):
    """One line of labels.jsonl for an article with cascade links."""
    def both(i):
        return m(record["source_sentences"][i]) | m(target(record, i))
    edits = direct(record)
    direct_m = set().union(*(both(j) for j in edits)) if edits else set()
    cascades = sorted({link["cascade"] for link in record["cascades"]})
    return {"article_id": record["id"], "direct": edits, "cascade_sentences": [
        {"sentence_idx": k, "label": "cascade", "shares_mention_with_direct": bool(both(k) & direct_m)}
        for k in cascades]}


def pairs(record, rng, neg_per_pos=NEG_PER_POS):
    keep = [i for i, entry in enumerate(record["labels"]) if entry["label"] == "KEEP"]
    out = []
    for link in record["cascades"]:
        j, k = link["direct"], link["cascade"]
        out.append({"article_id": record["id"], "j": j, "k": k, "label": 1})
        for n in rng.sample(keep, min(neg_per_pos, len(keep))):
            out.append({"article_id": record["id"], "j": j, "k": n, "label": 0})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="synthetic JSONL with cascade links")
    parser.add_argument("out", help="directory for labels.jsonl and pairs.jsonl")
    parser.add_argument("--eval-fraction", type=float, default=0.2, help="share of linked articles held out for evaluation")
    parser.add_argument("--neg-per-pos", type=int, default=NEG_PER_POS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    linked = [r for r in read_jsonl(args.data) if r.get("cascades")]
    rng = random.Random(args.seed)
    order = list(range(len(linked)))
    rng.shuffle(order)
    held = set(order[:round(args.eval_fraction * len(linked))])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = get_extractor()
    n_pairs = n_pos = shared = cascades = 0
    with open(out / "labels.jsonl", "w") as labels, open(out / "pairs.jsonl", "w") as train:
        for i, record in enumerate(linked):
            if i in held:
                row = label_row(record, m)
                labels.write(json.dumps(row) + "\n")
                cascades += len(row["cascade_sentences"])
                shared += sum(c["shares_mention_with_direct"] for c in row["cascade_sentences"])
            else:
                for pair in pairs(record, rng, args.neg_per_pos):
                    train.write(json.dumps(pair) + "\n")
                    n_pairs += 1
                    n_pos += pair["label"]
    print(f"{len(linked)} linked articles: {len(held)} for evaluation with {cascades} cascades "
          f"({shared} share a mention with a direct edit), {len(linked) - len(held)} for training "
          f"with {n_pairs} pairs ({n_pos} positive)")


if __name__ == "__main__":
    main()
