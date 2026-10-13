"""Rebuild the synthetic train set so that surface patterns do not give the labels away.

The first file (train_30k_diverse.jsonl) has four shortcuts:

    1. Every DROP sentence is quoted in a "Correction Notice" snippet, and no
       other snippet has that title.
    2. EDIT sentences have a snippet with the gold rewrite. KEEP sentences
       have no snippet, and all-KEEP articles get one contentless snippet.
    3. 28,000 of 36,982 EDIT sentences contain one of 12 fixed person names,
       whatever the replaced text was ("born in Elena Rostova, Chile").
    4. 280 EDIT sentences end in a fixed appended sentence ("As a result, the
       operation was classified as ...").

This script keeps the gold rewrites and rebuilds the rest:

    corruption  Each EDIT source is made again from its gold rewrite. One to
                three entities (spaCy en_core_web_sm, the types of
                datasets/collate.py) are replaced by an entity of the same
                type, word count and digit use drawn from the other gold rewrites. Years and integers
                move to a nearby year or an integer of the same length. A
                rewrite with no entity becomes KEEP.
    evidence    Every snippet has the title "Reference". Each KEEP and EDIT
                sentence gets a snippet with its gold text with probability
                P_SUPPORT, so a matching snippet does not separate KEEP from
                EDIT. Each DROP sentence gets a retraction notice from
                NOTICES. A share P_DECOY of KEEP sentences gets the same
                notice with the confirming verb. Snippets are shuffled.
    drop        A DROP sentence stays DROP with probability P_DROP. The others
                become EDIT: the sentence is the gold rewrite and the source
                is a corruption of it, as for every other EDIT.
    suffix      The appended sentences of shortcut 4 are removed.

Usage:
    python rebuild.py train_30k_diverse.jsonl train_30k_v2.jsonl
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict

MENTION_LABELS = {
    "CARDINAL", "DATE", "FAC", "GPE", "LOC", "MONEY", "NORP", "ORDINAL",
    "ORG", "PERCENT", "PERSON", "QUANTITY", "TIME",
}
SUFFIX = re.compile(r"\s*As a result, the operation was classified as [^.]*\.\s*$")
TITLE = "Reference"
P_SUPPORT = 0.8
P_DECOY = 0.15
P_DROP = 0.3
SWAPS = ((1, 0.6), (2, 0.3), (3, 0.1))
SNIPPET_CHARS = 300
# (frame, retracting verb phrase, confirming verb phrase)
NOTICES = (
    ("The claim that '{q}' was {v} following audit.", "retracted", "confirmed"),
    ("A later review {v} the statement '{q}'.", "withdrew", "upheld"),
    ("Editors have {v} this sentence: '{q}'", "removed", "retained"),
    ("'{q}' This statement is {v} by the current record.", "no longer supported", "still supported"),
    ("Record check: the passage '{q}' was found to be {v}.", "inaccurate", "accurate"),
    ("The sources for '{q}' were {v} on review.", "rejected", "accepted"),
    ("Following a review, the line '{q}' was {v}.", "struck from the article", "kept in the article"),
    ("'{q}' Reviewers marked this passage as {v}.", "unverified", "verified"),
)


def clip(text):
    text = " ".join(text.split())
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS].rsplit(" ", 1)[0]


def shape(text):
    if re.fullmatch(r"1\d{3}|20\d{2}", text):
        return "year"
    if re.fullmatch(r"\d+", text):
        return "int"
    return "text"


def pool_key(text, label):
    """Replacements share the type, the word count (up to 3) and the presence of digits."""
    return label, min(len(text.split()), 3), bool(re.search(r"\d", text))


def replacement(ent_text, label, sentence, pool, rng):
    kind = shape(ent_text)
    if kind == "year":
        return str(int(ent_text) + rng.choice([-1, 1]) * rng.randint(1, 15))
    if kind == "int":
        low = 10 ** (len(ent_text) - 1) if len(ent_text) > 1 else 1
        for _ in range(20):
            new = str(rng.randint(low, 10 ** len(ent_text) - 1))
            if new != ent_text:
                return new
        return None
    choices = pool[pool_key(ent_text, label)]
    lowered = sentence.lower()
    for _ in range(20):
        new = rng.choice(choices)
        if new.lower() != ent_text.lower() and new.lower() not in lowered:
            return new
    return None


def corrupt(target, ents, pool, rng):
    """The corrupted sentence and its stale spans, or None if nothing was replaced."""
    k = min(rng.choices([n for n, _ in SWAPS], [w for _, w in SWAPS])[0], len(ents))
    chosen = sorted(rng.sample(ents, k))
    parts, spans, cursor, length = [], [], 0, 0
    for start, end, label in chosen:
        new = replacement(target[start:end], label, target, pool, rng)
        if new is None:
            continue
        if not target[:start].strip() and new[0].islower():
            new = new[0].upper() + new[1:]
        parts += [target[cursor:start], new]
        length += start - cursor
        spans.append([length, length + len(new)])
        length += len(new)
        cursor = end
    if not spans:
        return None
    return "".join(parts) + target[cursor:], spans


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])

    with open(args.data) as f:
        records = [json.loads(line) for line in f if line.strip()]

    edits, stats = [], Counter()
    for record in records:
        for j, entry in enumerate(record["labels"]):
            if entry["label"] == "DROP" and rng.random() >= P_DROP:
                # the sentence is correct text of the first file, so it serves as a gold rewrite
                entry["label"], entry["target"] = "EDIT", record["source_sentences"][j]
                stats["drop_to_edit"] += 1
            if entry["label"] == "EDIT":
                entry["target"] = SUFFIX.sub("", entry["target"])
                edits.append((record, j))

    pool, found = defaultdict(list), []
    docs = nlp.pipe((record["labels"][j]["target"] for record, j in edits), batch_size=256, n_process=4)
    for doc in docs:
        ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents
                if e.label_ in MENTION_LABELS and "\n" not in e.text and e.text.strip() == e.text]
        found.append(ents)
        for start, end, label in ents:
            text = doc.text[start:end]
            pool[pool_key(text, label)].append(text)

    for (record, j), ents in zip(edits, found):
        entry = record["labels"][j]
        result = corrupt(entry["target"], ents, pool, rng) if ents else None
        if result is None:
            record["source_sentences"][j] = entry["target"]
            entry["label"], entry["stale_spans"] = "KEEP", []
            stats["edit_to_keep"] += 1
        else:
            record["source_sentences"][j], entry["stale_spans"] = result
            stats[f"swaps_{len(result[1])}"] += 1

    for record in records:
        evidence = []
        for sentence, entry in zip(record["source_sentences"], record["labels"]):
            if entry["label"] == "DROP":
                frame, verb, _ = rng.choice(NOTICES)
                evidence.append(frame.format(q=clip(sentence), v=verb))
                continue
            if rng.random() < P_SUPPORT:
                evidence.append(clip(entry["target"]))
            if entry["label"] == "KEEP" and rng.random() < P_DECOY:
                frame, _, verb = rng.choice(NOTICES)
                evidence.append(frame.format(q=clip(sentence), v=verb))
                stats["decoys"] += 1
        if not evidence:
            j = rng.choice([j for j, e in enumerate(record["labels"]) if e["label"] != "DROP"])
            evidence.append(clip(record["labels"][j]["target"]))
        rng.shuffle(evidence)
        record["evidence"] = [{"title": TITLE, "text": text} for text in evidence]
        record["target_sentences"] = [e["target"] for e in record["labels"] if e["label"] != "DROP"]
        record["target"] = " ".join(record["target_sentences"])
        stats.update(e["label"] for e in record["labels"])

    with open(args.out, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
