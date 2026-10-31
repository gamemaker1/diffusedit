"""Task files for the evidence writers (prompt.md) from generated records.

Each line of a task file is one article, {"id", "sentences": [...]}, with one
entry {"i", "label", "text", "fixes"} per sentence. text is the correct
sentence, or for DROP the sentence that no longer holds. fixes lists the
{"wrong", "right"} pairs of an EDIT sentence.

The cascade sentence k of a link {"direct": j, "cascade": k} gets the label
CASCADE. Its fixes are listed so that no passage holds a wrong string, and
"avoid" lists its right strings that the fixes of j do not contain. No
passage may state k's fact, so the cascade can be found only through j.

Usage:
    python tasks.py regen/generated.jsonl regen/in --chunk 25
"""

import argparse
import json
import re
from pathlib import Path

from merge import gold_spans


def fixes(record, i):
    entry = record["labels"][i]
    return [{"wrong": w, "right": r} for w, r in gold_spans(record["source_sentences"][i], entry["target"], entry["stale_spans"])]


def cascade_avoid(record):
    """{k: right strings of cascade k that no right string of its direct edit j contains as whole words}."""
    out = {}
    for link in record.get("cascades", []):
        direct = [f["right"].lower() for f in fixes(record, link["direct"])]
        out[link["cascade"]] = sorted({f["right"] for f in fixes(record, link["cascade"])
                                       if not any(re.search(rf"(?<!\w){re.escape(f['right'].lower())}(?!\w)", d)
                                                  for d in direct)})
    return out


def task(record):
    avoid = cascade_avoid(record)
    sentences = []
    for i, (source, entry) in enumerate(zip(record["source_sentences"], record["labels"])):
        label = "CASCADE" if i in avoid else entry["label"]
        item = {"i": i, "label": label, "text": source if label == "DROP" else entry["target"]}
        if label in ("EDIT", "CASCADE"):
            item["fixes"] = fixes(record, i)
        if label == "CASCADE":
            item["avoid"] = avoid[i]
        sentences.append(item)
    return {"id": record["id"], "sentences": sentences}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("out", help="directory for chunk_NNNN.jsonl")
    parser.add_argument("--chunk", type=int, default=25, help="articles per task file")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in open(args.data) if line.strip()]
    for c in range(0, len(records), args.chunk):
        with open(out / f"chunk_{c // args.chunk:04d}.jsonl", "w") as f:
            for record in records[c:c + args.chunk]:
                f.write(json.dumps(task(record), ensure_ascii=False) + "\n")
    print(f"{len(records)} articles in {(len(records) + args.chunk - 1) // args.chunk} task files")


if __name__ == "__main__":
    main()
