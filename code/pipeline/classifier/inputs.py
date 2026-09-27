"""Build LLaDA input sequences with sentence token spans from collated records.

A sequence is [BOS] + evidence + separator + article, the E ⊕ A input of
Algorithm 1. Each article sentence is tokenized on its own, so its token span
is exact and mean pooling (Eq. 4) covers that sentence and nothing else.

Check spans against a split:
    python -m classifier.inputs ../datasets/collated/test.jsonl --n 50
"""

import argparse
import json
import random
from dataclasses import dataclass

import numpy as np

LABELS = ("KEEP", "EDIT", "DROP")
LABEL_ID = {name: i for i, name in enumerate(LABELS)}
KEEP, EDIT, DROP = range(3)

MODEL = "GSAI-ML/LLaDA-8B-Base"
EVIDENCE_BUDGET = 600
SEPARATOR = "\n\n"


@dataclass
class Example:
    id: str
    input_ids: np.ndarray
    spans: list[tuple[int, int]]
    labels: list[int]


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def encode(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def evidence_ids(tok, evidence, budget=EVIDENCE_BUDGET):
    """Snippets as "title: text", one per line, cut at the token budget.

    The collated data is already truncated to 600 LLaDA tokens without titles.
    The cut here enforces the budget again after titles are added.
    """
    ids = []
    for i, snippet in enumerate(evidence):
        text = f"{snippet['title']}: {snippet['text']}"
        ids += encode(tok, text if i == 0 else "\n" + text)
    return ids[:budget]


def build(record, tok):
    ids = [] if tok.bos_token_id is None else [tok.bos_token_id]
    ids += evidence_ids(tok, record["evidence"])
    ids += encode(tok, SEPARATOR)

    spans = []
    for i, sentence in enumerate(record["source_sentences"]):
        sentence = sentence.strip()
        piece = encode(tok, sentence if i == 0 else " " + sentence)
        if not piece:
            raise ValueError(f"{record['id']}: sentence {i} has no tokens")
        spans.append((len(ids), len(ids) + len(piece)))
        ids += piece

    labels = [LABEL_ID[entry["label"]] for entry in record["labels"]]
    if len(labels) != len(spans):
        raise ValueError(f"{record['id']}: {len(labels)} labels for {len(spans)} sentences")
    return Example(record["id"], np.asarray(ids, dtype=np.int32), spans, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--n", type=int, default=50, help="sentences to check")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    records = list(read_jsonl(args.data))
    examples = [build(r, tok) for r in records]

    lengths = np.array([len(e.input_ids) for e in examples])
    print(f"{len(examples)} articles, sequence tokens mean {lengths.mean():.0f}, "
          f"p90 {np.percentile(lengths, 90):.0f}, max {lengths.max()}")

    rng = random.Random(args.seed)
    pairs = [(r, e, j) for r, e in zip(records, examples) for j in range(len(e.spans))]
    mismatches = 0
    for record, example, j in rng.sample(pairs, min(args.n, len(pairs))):
        start, end = example.spans[j]
        decoded = tok.decode(example.input_ids[start:end].tolist())
        expected = record["source_sentences"][j]
        if "".join(decoded.split()) != "".join(expected.split()):
            mismatches += 1
            print(f"MISMATCH {example.id} [{j}]\n  expected: {expected!r}\n  decoded:  {decoded!r}")
    print(f"{mismatches} of {min(args.n, len(pairs))} sampled spans mismatch")


if __name__ == "__main__":
    main()
