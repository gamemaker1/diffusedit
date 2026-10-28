"""Split the synthetic set into train, val and test by seed article.

Records made from the same seed paragraph share their first sentence, taken
from the gold text, or from the source when the sentence is a DROP, which the
generator does not alter. 20 of the 9183 records repeat the seed of an earlier
record. All records of one seed go to the same split, so no test article has
a near copy in train. Val and test are held out for evaluation, and no job
runs on them during development.

Usage:
    python split.py train.jsonl split
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def seed_key(record):
    first = record["labels"][0]["target"] or record["source_sentences"][0]
    return " ".join(re.findall(r"\w+", first.lower()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("out", help="directory for train.jsonl, val.jsonl and test.jsonl")
    parser.add_argument("--val", type=float, default=0.1, help="fraction of seeds in val")
    parser.add_argument("--test", type=float, default=0.1, help="fraction of seeds in test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    groups = defaultdict(list)
    for line in open(args.data):
        if line.strip():
            record = json.loads(line)
            groups[seed_key(record)].append(record)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    n_test, n_val = round(len(keys) * args.test), round(len(keys) * args.val)
    splits = {"test": keys[:n_test], "val": keys[n_test:n_test + n_val], "train": keys[n_test + n_val:]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        records = sorted((r for k in members for r in groups[k]), key=lambda r: r["id"])
        with open(out / f"{name}.jsonl", "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        labels = [entry["label"] for r in records for entry in r["labels"]]
        counts = {label: labels.count(label) for label in ("KEEP", "EDIT", "DROP")}
        print(f"{name}: {len(members)} seeds, {len(records)} articles, {counts}")


if __name__ == "__main__":
    main()
