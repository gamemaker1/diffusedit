"""Cascade labels and consequence-head pairs from synthetic cascade links.

A synthetic record lists its injected pairs in "cascades", one
{"direct": j, "cascade": k} link each, so the cascades of Section 8.5 are
known without hand labels or mining. The direct edits of such an article are
its EDIT and DROP sentences that are not the cascade of a link. A record
without links has no cascades, and all its EDIT and DROP sentences are
direct edits.

A link whose cascade fix (the correct strings of k that the fix of j does
not contain) also occurs in another sentence of the outdated article is left
out of both files, because the article itself states the fix. Generators
before that rule produced about 13% such links.

A seeded fraction of the articles with a link becomes the evaluation set and
the rest give training pairs, so no article is in both. With --heads-holdout,
the evaluation set is instead the linked articles among those that
classifier.train and staleness.train hold out with the same fraction and seed
(classifier.train.split_by_article over the file order), so the loop and the
detectors are scored on articles no head was fitted on. The heads still used
those articles for early stopping and for tuning tau_d and tau, so the scores
are development numbers, and final numbers come from the test split.

    labels.jsonl  the format of cascading.evaluate, one line per evaluation
                  article, with its direct edits and its cascade sentences.
                  shares_mention_with_direct follows datasets/collate.py.
                  It is true when the mentions of both versions of k meet
                  the mentions of both versions of some direct edit. Each
                  cascade sentence also carries the kind of its link.
    loop_ids.json the evaluation article ids in an order that cycles through
                  the cascade kinds, shuffled within each kind, so that the
                  first N ids for infilling.loop cover every kind.
    pairs.jsonl   one ordered pair per line, {"article_id", "j", "k", "label"}.
                  Each link (j, k) is a positive, and each positive draws up
                  to NEG_PER_POS KEEP sentences k' of its article as negative
                  pairs (j, k'), as in Section 6.4. Each positive also draws
                  up to HARD_PER_POS hard negatives, sentences k' that hold
                  a cue word of its kind: a wrong or right string of some
                  training cascade of that kind, such as a country name or a
                  pronoun. k' is a KEEP sentence or a direct EDIT other than
                  j, so it is correct or wrong for its own reason and not
                  because of j. They come from its own article first and
                  then from the other training articles, paired with a
                  direct EDIT of their article. Without them a cue word
                  alone would separate the positives from the negatives.

Usage:
    python -m cascading.links ../datasets/synthetic/cascade/source.jsonl ../datasets/synthetic/cascade
"""

import argparse
import json
import random
import re
from pathlib import Path

from cascading.mentions import get_extractor
from classifier.inputs import read_jsonl

NEG_PER_POS = 3
HARD_PER_POS = 3


def target(record, i):
    """The gold rewrite of source sentence i, or "" for a deleted sentence."""
    entry = record["labels"][i]
    return entry["target"] if entry["label"] != "DROP" else ""


def fixes(record, i):
    """(wrong, right) strings of each stale span of sentence i: the target text between unchanged neighbours."""
    source, entry = record["source_sentences"][i], record["labels"][i]
    target, spans = entry["target"], entry["stale_spans"]
    out, pos_s, pos_t = [], 0, 0
    for n, (a, b) in enumerate(spans):
        pos_t += a - pos_s
        after = source[b:spans[n + 1][0]] if n + 1 < len(spans) else source[b:]
        end = len(target) - len(after) if n + 1 == len(spans) else target.index(after, pos_t) if after else pos_t
        out.append((source[a:b], target[pos_t:end]))
        pos_s, pos_t = b, end
    return out


def whole(word, text):
    return re.search(rf"(?<!\w){re.escape(word.lower())}(?!\w)", text.lower()) is not None


def fix_in_article(record, link):
    """True when a fix string of the cascade, absent from the direct fix, occurs in another sentence."""
    j, k = link["direct"], link["cascade"]
    direct_fix = " ".join(right for _, right in fixes(record, j))
    others = " ".join(s for i, s in enumerate(record["source_sentences"]) if i != k)
    return any(whole(right, others) for _, right in fixes(record, k) if not whole(right, direct_fix))


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
    kinds = {link["cascade"]: link.get("kind") for link in record["cascades"]}
    return {"article_id": record["id"], "direct": edits, "cascade_sentences": [
        {"sentence_idx": k, "label": "cascade", "kind": kinds[k], "shares_mention_with_direct": bool(both(k) & direct_m)}
        for k in sorted(kinds)]}


def cycle_kinds(rows, rng):
    """Article ids of label rows, taking one article of each kind in turn."""
    by_kind = {}
    for row in rows:
        by_kind.setdefault(row["cascade_sentences"][0].get("kind"), []).append(row["article_id"])
    queues = [by_kind[k] for k in sorted(by_kind, key=str)]
    for q in queues:
        rng.shuffle(q)
    order = []
    while any(queues):
        order += [q.pop() for q in queues if q]
    return order


def keep(record):
    return [i for i, entry in enumerate(record["labels"]) if entry["label"] == "KEEP"]


def pairs(record, rng, neg_per_pos=NEG_PER_POS):
    out = []
    for link in record["cascades"]:
        j, k = link["direct"], link["cascade"]
        out.append({"article_id": record["id"], "j": j, "k": k, "label": 1})
        for n in rng.sample(keep(record), min(neg_per_pos, len(keep(record)))):
            out.append({"article_id": record["id"], "j": j, "k": n, "label": 0})
    return out


def cue_words(records):
    """{kind: the lowercased wrong and right strings of the cascades of that kind}."""
    words = {}
    for record in records:
        for link in record["cascades"]:
            for wrong, right in fixes(record, link["cascade"]):
                words.setdefault(link.get("kind"), set()).update({wrong.lower(), right.lower()})
    return {kind: re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(sorted(map(re.escape, w), key=len, reverse=True)))
            for kind, w in words.items()}


def hard_negatives(train, cues, rng, hard_per_pos=HARD_PER_POS, taken=()):
    """Negative pairs (j, k') whose k' holds a cue word of a positive's kind, none in `taken`.

    k' is a KEEP sentence or a direct EDIT, and j a direct EDIT other than k'.
    """
    def unrelated(record):
        cascades = {link["cascade"] for link in record["cascades"]}
        return [i for i, entry in enumerate(record["labels"]) if entry["label"] in ("KEEP", "EDIT") and i not in cascades]

    pools = {}
    for record in train:
        edits = [i for i in direct(record) if record["labels"][i]["label"] == "EDIT"]
        for kind, cue in cues.items():
            for n in unrelated(record):
                others = [e for e in edits if e != n]
                if others and cue.search(record["source_sentences"][n].lower()):
                    pools.setdefault(kind, []).append((record, n, others))
    for pool in pools.values():
        rng.shuffle(pool)
    # Each pool entry is drawn at most once, in its shuffled order.
    pointer = dict.fromkeys(pools, 0)
    used, out = {(p["article_id"], p["j"], p["k"]) for p in taken}, []
    for record in train:
        for link in record["cascades"]:
            kind, j = link.get("kind"), link["direct"]
            candidates = [(record, n, j) for n in unrelated(record)
                          if n != j and cues[kind].search(record["source_sentences"][n].lower())]
            pool = pools.get(kind, [])
            drawn = 0
            while drawn < hard_per_pos and (candidates or pointer[kind] < len(pool)):
                if candidates:
                    other, n, d = candidates.pop(0)
                else:
                    other, n, edits = pool[pointer[kind]]
                    pointer[kind] += 1
                    d = rng.choice(edits)
                if (other["id"], d, n) in used:
                    continue
                used.add((other["id"], d, n))
                out.append({"article_id": other["id"], "j": d, "k": n, "label": 0})
                drawn += 1
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="synthetic JSONL with cascade links")
    parser.add_argument("out", help="directory for labels.jsonl and pairs.jsonl")
    parser.add_argument("--eval-fraction", type=float, default=0.2, help="share of linked articles held out for evaluation")
    parser.add_argument("--neg-per-pos", type=int, default=NEG_PER_POS)
    parser.add_argument("--hard-per-pos", type=int, default=HARD_PER_POS)
    parser.add_argument("--heads-holdout", type=float, help="hold out the heads' held-out articles, this fraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = list(read_jsonl(args.data))
    excluded = 0
    for record in records:
        links = record.get("cascades", [])
        record["cascades"] = [link for link in links if not fix_in_article(record, link)]
        excluded += len(links) - len(record["cascades"])
    print(f"{excluded} links left out because the outdated article states the cascade fix elsewhere")
    rng = random.Random(args.seed)
    linked = [r for r in records if r.get("cascades")]
    if args.heads_holdout:
        import numpy as np
        n = len(records)
        chosen = set(np.random.default_rng(args.seed).choice(n, size=max(1, round(args.heads_holdout * n)), replace=False).tolist())
        held_ids = {records[a]["id"] for a in chosen}
    else:
        order = list(range(len(linked)))
        rng.shuffle(order)
        held_ids = {linked[i]["id"] for i in order[:round(args.eval_fraction * len(linked))]}
    held = {i for i, r in enumerate(linked) if r["id"] in held_ids}
    # Hard negatives come from every article outside the held-out set, linked or not.
    train_records = [r for r in records if r["id"] not in held_ids]
    cues = cue_words([r for r in linked if r["id"] not in held_ids])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = get_extractor()
    n_pairs = n_pos = shared = cascades = 0
    rows = []
    with open(out / "labels.jsonl", "w") as labels, open(out / "pairs.jsonl", "w") as train:
        for i, record in enumerate(linked):
            if i in held:
                row = label_row(record, m)
                rows.append(row)
                labels.write(json.dumps(row) + "\n")
                cascades += len(row["cascade_sentences"])
                shared += sum(c["shares_mention_with_direct"] for c in row["cascade_sentences"])
        train_pairs = [pair for i, record in enumerate(linked) if i not in held for pair in pairs(record, rng, args.neg_per_pos)]
        if args.hard_per_pos:
            train_pairs += hard_negatives([r for i, r in enumerate(linked) if i not in held] +
                                          [r for r in train_records if not r.get("cascades")], cues, rng, args.hard_per_pos,
                                          taken=train_pairs)
        for pair in train_pairs:
            train.write(json.dumps(pair) + "\n")
            n_pairs += 1
            n_pos += pair["label"]
    (out / "loop_ids.json").write_text(json.dumps(cycle_kinds(rows, random.Random(args.seed))))
    print(f"{len(linked)} linked articles: {len(held)} for evaluation with {cascades} cascades "
          f"({shared} share a mention with a direct edit), {len(linked) - len(held)} for training "
          f"with {n_pairs} pairs ({n_pos} positive)")
    by_id = {r["id"]: r for r in records}
    for kind, cue in sorted(cues.items(), key=lambda x: str(x[0])):
        hits = [pair["label"] for pair in train_pairs if cue.search(by_id[pair["article_id"]]["source_sentences"][pair["k"]].lower())]
        positives = sum(1 for r in linked if r["id"] not in held_ids for link in r["cascades"] if link.get("kind") == kind)
        print(f"  {kind}: {positives} positives, {len(hits)} pairs whose k holds a cue word, "
              f"{sum(hits) / max(1, len(hits)):.2f} of them positive")


if __name__ == "__main__":
    main()
