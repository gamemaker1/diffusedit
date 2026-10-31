"""Stress-test data for the repair step, on evidence that breaks the rules the
synthetic generator guarantees, and on real FRUIT updates.

The generator puts every new value of an EDIT sentence in the evidence and no
old value anywhere in it. The evidence veto, the rejection of unsupported
names and the evidence-copy candidates of infilling.repair rely on these two
rules. Each data set here breaks one of them on the 200 fresh val articles:

    missing    for half of the EDIT sentences, drawn with seed 0, every
               passage that holds a new value of the sentence is removed.
               infilling.evaluate counts such a sentence as unsupported.
    oldvalue   one passage per EDIT sentence is added at a seeded position. It
               holds the old values of the sentence in another context and
               none of its new values. The passages are written from the task
               files of `tasks`.
    fruit      200 FRUIT train articles with an EDIT sentence that has stale
               spans, drawn with seed 42, unchanged. FRUIT val and test stay
               untouched.

Each writes <out>/<name>.jsonl and <out>/<name>_ids.json, and the entity file
of infilling.ner must then be built for it.

    python -m experiments.stress missing VAL IDS OUT
    python -m experiments.stress tasks VAL IDS TASKDIR --chunk 50
    python -m experiments.stress oldvalue VAL IDS PASSAGEDIR OUT
    python -m experiments.stress fruit FRUIT_TRAIN OUT
"""

import argparse
import json
import random
import re
from pathlib import Path


def read(path, ids=None):
    out = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if ids is None or record["id"] in ids:
                out[record["id"]] = record
    return out


def gold_spans(source, target, spans):
    """(old, new) text of every stale span, the target text between the unchanged neighbours."""
    out, pos_s, pos_t = [], 0, 0
    for k, (a, b) in enumerate(spans):
        pos_t += a - pos_s
        after = source[b:spans[k + 1][0]] if k + 1 < len(spans) else source[b:]
        end = len(target) - len(after) if k + 1 == len(spans) else target.index(after, pos_t) if after else pos_t
        out.append((source[a:b], target[pos_t:end]))
        pos_s, pos_t = b, end
    return out


def pairs(record, j):
    entry = record["labels"][j]
    return gold_spans(record["source_sentences"][j], entry["target"], entry["stale_spans"])


def holds(text, value):
    """Whether text holds value as whole words, ignoring case."""
    value = value.strip()
    return bool(value) and re.search(rf"(?<!\w){re.escape(value.lower())}(?!\w)", text.lower()) is not None


def full(passage):
    return f"{passage['title']} {passage['text']}"


def edits(records):
    return [(i, j) for i in sorted(records) for j, e in enumerate(records[i]["labels"]) if e["label"] == "EDIT"]


def write(records, order, out, name, extra=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.jsonl", "w") as f:
        for i in order:
            f.write(json.dumps(records[i]) + "\n")
    (out / f"{name}_ids.json").write_text(json.dumps(sorted(order)))
    if extra is not None:
        (out / f"{name}_sentences.json").write_text(json.dumps(extra))


def missing(args):
    records = read(args.data, set(json.loads(Path(args.ids).read_text())))
    pool = edits(records)
    chosen = sorted(random.Random(0).sample(pool, len(pool) // 2))
    removed = 0
    for i, j in chosen:
        new = [n for _, n in pairs(records[i], j) if n.strip()]
        before = records[i]["evidence"]
        records[i]["evidence"] = [p for p in before if not any(holds(full(p), n) for n in new)]
        removed += len(before) - len(records[i]["evidence"])
    write(records, sorted(records), args.out, "missing", [list(c) for c in chosen])
    print(f"{len(records)} articles, {len(pool)} EDIT sentences, {len(chosen)} chosen, {removed} passages removed")


def tasks(args):
    records = read(args.data, set(json.loads(Path(args.ids).read_text())))
    out = Path(args.taskdir)
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(records)
    for c in range(0, len(ids), args.chunk):
        with open(out / f"chunk_{c // args.chunk:02d}.jsonl", "w") as f:
            for i in ids[c:c + args.chunk]:
                r = records[i]
                sentences = [{"i": j, "outdated": r["source_sentences"][j], "updated": r["labels"][j]["target"],
                              "old": [o for o, _ in pairs(r, j) if o.strip()],
                              "new": [n for _, n in pairs(r, j) if n.strip()]}
                             for j, e in enumerate(r["labels"]) if e["label"] == "EDIT"]
                f.write(json.dumps({"id": i, "subject": r["source_sentences"][0], "edits": sentences}) + "\n")
    print(f"{len(ids)} articles in {(len(ids) + args.chunk - 1) // args.chunk} task files")


def shingles(text, n=7):
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[k:k + n]) for k in range(len(words) - n + 1)}


def check(record, j, passage):
    """None if the added passage of EDIT sentence j is valid, else the reason."""
    if not isinstance(passage, dict) or not passage.get("title") or not passage.get("text"):
        return "malformed"
    text = full(passage)
    old = [o for o, _ in pairs(record, j) if o.strip()]
    new = [n for _, n in pairs(record, j) if n.strip()]
    if not all(holds(text, o) for o in old):
        return "an old value is missing"
    if any(holds(text, n) for n in new if not any(holds(n, o) for o in old)):
        return "holds a new value"
    copied = set().union(*(shingles(s) for s in record["source_sentences"] + [record["labels"][j]["target"]]))
    # an old value of seven or more words has to appear, so its own runs do not count as copying
    copied -= set().union(*(shingles(o) for o in old))
    if shingles(text) & copied:
        return "copies seven words of the article"
    if not 10 <= len(passage["text"].split()) <= 80:
        return "length"
    return None


def oldvalue(args):
    records = read(args.data, set(json.loads(Path(args.ids).read_text())))
    written = {}
    for path in sorted(Path(args.passages).glob("*.jsonl")):
        for line in open(path):
            line = line.strip()
            if line:
                row = json.loads(line)
                written[(row["id"], int(row["i"]))] = row
    rng = random.Random(0)
    bad, added = {}, 0
    for i, j in edits(records):
        row = written.get((i, j))
        reason = "not written" if row is None else check(records[i], j, row)
        if reason:
            bad[f"{i}:{j}"] = reason
            continue
        evidence = records[i]["evidence"]
        evidence.insert(rng.randint(0, len(evidence)), {"title": row["title"], "text": row["text"]})
        added += 1
    write(records, sorted(records), args.out, "oldvalue")
    Path(args.out, "oldvalue_rejected.json").write_text(json.dumps(bad, indent=1))
    print(f"{len(records)} articles, {added} passages added, {len(bad)} sentences without one")
    for k, v in list(bad.items())[:20]:
        print(" ", k, v)


def fruit(args):
    pool = []
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            if any(e["label"] == "EDIT" and e.get("stale_spans") for e in r["labels"]):
                pool.append(line)
    rows = random.Random(42).sample(pool, 200)
    records = {json.loads(l)["id"]: json.loads(l) for l in rows}
    write(records, sorted(records), args.out, "fruit")
    print(f"{len(pool)} FRUIT train articles with a stale EDIT sentence, 200 sampled")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("missing")
    p.add_argument("data"); p.add_argument("ids"); p.add_argument("out")
    p = sub.add_parser("tasks")
    p.add_argument("data"); p.add_argument("ids"); p.add_argument("taskdir")
    p.add_argument("--chunk", type=int, default=50)
    p = sub.add_parser("oldvalue")
    p.add_argument("data"); p.add_argument("ids"); p.add_argument("passages"); p.add_argument("out")
    p = sub.add_parser("fruit")
    p.add_argument("data"); p.add_argument("out")
    args = parser.parse_args()
    {"missing": missing, "tasks": tasks, "oldvalue": oldvalue, "fruit": fruit}[args.command](args)


if __name__ == "__main__":
    main()
