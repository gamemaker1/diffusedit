"""Oracle-repaired views of a dataset, for the hidden states of the cascade detectors.

The detectors compare the state of a sentence before a round's repairs with
its state after them. Offline, without the edit loop, the repairs are the
gold rewrites of the direct edits (cascading.links.direct). Two views of each
record go to classifier.extract.

    after     the source article with every direct EDIT replaced by its gold
              rewrite. A direct DROP keeps its source text, because an empty
              sentence has no tokens and its row would be undefined.
    evidence  the after view with the rewritten sentences appended to the
              evidence as snippets (EditedAsEvidenceDetector.augment_evidence),
              the input of the edited-as-evidence detector in round 2.

Sentence counts and order stay as in the source, so row i of the source
features and row i of a view's features hold the same sentence.

Usage:
    python -m cascading.views ../datasets/synthetic/cascades.jsonl ../datasets/synthetic/views
"""

import argparse
import json
from pathlib import Path

from cascading.detectors import EditedAsEvidenceDetector
from cascading.links import direct, target
from classifier.inputs import read_jsonl


def after(record):
    """The record with each direct EDIT sentence replaced by its gold rewrite, and the rewrites."""
    sentences = list(record["source_sentences"])
    rewrites = {}
    for j in direct(record):
        if record["labels"][j]["label"] == "EDIT" and target(record, j).strip():
            sentences[j] = rewrites[j] = target(record, j)
    return {**record, "source_sentences": sentences}, rewrites


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("out", help="directory for after.jsonl and evidence.jsonl")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    augment = EditedAsEvidenceDetector().augment_evidence
    n = rewritten = 0
    with open(out / "after.jsonl", "w") as a, open(out / "evidence.jsonl", "w") as e:
        for record in read_jsonl(args.data):
            view, rewrites = after(record)
            a.write(json.dumps(view) + "\n")
            e.write(json.dumps({**view, "evidence": augment(record["evidence"], rewrites)}) + "\n")
            n += 1
            rewritten += len(rewrites)
    print(f"{n} articles, {rewritten} direct edits rewritten")


if __name__ == "__main__":
    main()
