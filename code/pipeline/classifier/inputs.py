"""Build LLaDA input sequences with sentence token spans from collated records.

A sequence is [BOS] + evidence + separator + article, the E ⊕ A input of
Algorithm 1. Each article sentence is tokenized on its own, so its token span
is exact and mean pooling (Eq. 4) covers that sentence and nothing else.

Every EDIT sentence also gets one stale label per token (Section 8.4). A token
is stale when its character range overlaps a stale span of the label.

Check spans and stale tokens against a split:
    python -m classifier.inputs ../datasets/collated/train.jsonl --n 50
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
    stale: list[np.ndarray | None]


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


def stale_marks(offsets, stale_spans, shift):
    """1 for each token whose character range overlaps a stale span, else 0."""
    marks = np.zeros(len(offsets), dtype=np.int8)
    for t, (start, end) in enumerate(offsets):
        start, end = start - shift, end - shift
        if any(start < b and end > a for a, b in stale_spans):
            marks[t] = 1
    return marks


def build(record, tok):
    ids = [] if tok.bos_token_id is None else [tok.bos_token_id]
    ids += evidence_ids(tok, record["evidence"])
    ids += encode(tok, SEPARATOR)

    if len(record["labels"]) != len(record["source_sentences"]):
        raise ValueError(f"{record['id']}: {len(record['labels'])} labels for "
                         f"{len(record['source_sentences'])} sentences")
    spans, labels, stale = [], [], []
    for i, (sentence, entry) in enumerate(zip(record["source_sentences"], record["labels"])):
        lead = len(sentence) - len(sentence.lstrip())
        prefix = "" if i == 0 else " "
        enc = tok(prefix + sentence.strip(), add_special_tokens=False, return_offsets_mapping=True)
        piece = enc["input_ids"]
        if not piece:
            raise ValueError(f"{record['id']}: sentence {i} has no tokens")
        spans.append((len(ids), len(ids) + len(piece)))
        ids += piece
        labels.append(LABEL_ID[entry["label"]])
        if entry["label"] == "EDIT":
            stale.append(stale_marks(enc["offset_mapping"], entry["stale_spans"], len(prefix) - lead))
        else:
            stale.append(None)
    return Example(record["id"], np.asarray(ids, dtype=np.int32), spans, labels, stale)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--n", type=int, default=50, help="sentences to check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="read only the first N articles")
    args = parser.parse_args()

    import itertools
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    records = list(itertools.islice(read_jsonl(args.data), args.limit))
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

    edits = [(r, e, j) for r, e, j in pairs if e.stale[j] is not None]
    marked = sum(int(e.stale[j].sum()) for _, e, j in edits)
    total = sum(len(e.stale[j]) for _, e, j in edits)
    print(f"{len(edits)} EDIT sentences, {total} tokens, {marked} stale ({marked / max(total, 1):.1%})")
    for record, example, j in rng.sample(edits, min(5, len(edits))):
        start, _ = example.spans[j]
        sentence = record["source_sentences"][j]
        stale_text = [sentence[a:b] for a, b in record["labels"][j]["stale_spans"]]
        tokens = [tok.decode([int(example.input_ids[start + t])]) for t in np.flatnonzero(example.stale[j])]
        print(f"{example.id} [{j}]\n  stale spans:  {stale_text}\n  stale tokens: {tokens}")


if __name__ == "__main__":
    main()
