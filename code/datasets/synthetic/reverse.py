"""Reverse the direction of the synthetic edits, so the outdated article is the real one.

The generator writes wrong values into a real Wikipedia lead and keeps the
lead as the gold target. LLaDA has read such leads in pretraining, so a
wrong value often clashes with what it knows, and a head or an infilling run
finds the stale text without the evidence. Here the real sentence becomes
the source and the mutated sentence the target:

    EDIT      source = the old target sentence, target = the old source
              sentence, and the stale spans are the old gold spans, now in
              the source.
    KEEP      unchanged.
    DROP      unchanged. The sentence is already a real sentence that the
              target deletes, and the evidence already retracts it.
    evidence  every old gold string of a direct edit is replaced, as a whole
              word, by its mutated string, so the evidence states the new
              values. Cascade sentences have no evidence of their own.

These changes are undone, so the sentence keeps its real text:

    names       the first name that the gender kind gave the subject
    ambiguous   a gender link whose direct edit turns "actress" into "actor",
                which is also used for women, so the new pronouns would not
                follow from it. The direct edit and its cascades are undone.
    conflict    a real string that two sentences mutate into different
                strings. The later sentence's change is undone.

    listed      a link listed in --undo, so that a kind does not make its cue
                words (country names, demonyms, occupations) stale far more
                often than other words.

A link whose direct or cascade sentence loses a span this way is removed,
and its cascade sentences are undone with it.

The evidence replacement then leaves a record out when a check fails:

    insertion   a stale span whose old gold text is empty, which has no
                source token to mark
    shared      an old gold string of a direct edit occurs in a source
                sentence that does not change it, so the replacement would
                also rewrite the evidence of that sentence
    unsupported a mutated string of a direct edit is not in the new evidence
    leftover    an old gold string of a direct edit is still in the evidence
    cascade     a mutated string of a cascade sentence that its direct edit
                does not contain is in the evidence (pronouns excepted, which
                the evidence uses for other people)

--flip-only writes every record that is not an insertion, with its evidence
left as it was, for writers (prompt.md) to give new evidence through
tasks.py and merge.py.

Usage:
    python reverse.py split/train.jsonl reversed/train.jsonl --rejects reversed/train_rejects.json
    python reverse.py rejected.jsonl flipped.jsonl --flip-only
    python reverse.py split/train.jsonl reversed/train.jsonl --undo undo.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from synthetic.mutators.cascades import FIRST_NAMES  # noqa: E402

NAMES = set(FIRST_NAMES["M"]) | set(FIRST_NAMES["F"])
PRONOUNS = {"he", "him", "his", "himself", "she", "her", "hers", "herself"}
NEUTRAL = {"actor"}


def edits(source, target, spans):
    """(a, b, c, d) per stale span: source[a:b] became target[c:d]."""
    out, pos_s, pos_t = [], 0, 0
    for n, (a, b) in enumerate(spans):
        pos_t += a - pos_s
        after = source[b:spans[n + 1][0]] if n + 1 < len(spans) else source[b:]
        end = len(target) - len(after) if n + 1 == len(spans) else target.index(after, pos_t) if after else pos_t
        out.append((a, b, pos_t, end))
        pos_s, pos_t = b, end
    return out


def word(s):
    return re.compile(rf"(?<!\w){re.escape(s)}(?!\w)", re.IGNORECASE)


def has(s, text):
    return word(s).search(text) is not None


def same_case(new, old):
    if old.isupper() and len(old) > 1:
        return new.upper()
    if old[:1].isupper():
        return new[:1].upper() + new[1:]
    if old[:1].islower() and new[:1].isupper() and not new.isupper() and new.lower() in PRONOUNS:
        return new.lower()
    return new


def flip(record, listed=()):
    """(reversed record with the old evidence, changes {i: [(old, new)]}) or (None, "insertion").

    listed holds (direct, cascade) pairs of links to undo.
    """
    gender = any(link.get("kind") == "gender" for link in record.get("cascades", []))
    spans = {}
    for i, (source, entry) in enumerate(zip(record["source_sentences"], record["labels"])):
        if entry["label"] != "EDIT":
            continue
        kept = []
        for a, b, c, d in edits(source, entry["target"], entry["stale_spans"]):
            old, new = entry["target"][c:d], source[a:b]
            if not old.strip():
                return None, "insertion"
            if gender and new in NAMES and old[:1].isupper() and " " not in old and old.lower() not in PRONOUNS:
                continue
            kept.append((c, d, old, new))
        spans[i] = kept

    links = list(record.get("cascades", []))
    undo = {end for link in links if (link["direct"], link["cascade"]) in set(map(tuple, listed))
            for end in (link["direct"], link["cascade"])}
    for link in links:
        news = [new.lower() for _, _, _, new in spans.get(link["direct"], [])]
        if link.get("kind") == "gender" and NEUTRAL & set(news):
            undo |= {link["direct"], link["cascade"]}
    mapping = {}
    for i in sorted(spans):
        if i in undo:
            continue
        for _, _, old, new in spans[i]:
            if mapping.setdefault(old.lower(), new.lower()) != new.lower():
                undo.add(i)
    # A link loses both ends when either end is undone, until nothing changes.
    while True:
        more = {end for link in links if link["direct"] in undo or link["cascade"] in undo
                for end in (link["direct"], link["cascade"]) if end not in undo}
        if not more:
            break
        undo |= more
    links = [link for link in links if link["direct"] not in undo and link["cascade"] not in undo
             and spans.get(link["direct"]) and spans.get(link["cascade"])]

    sources, labels, changes = [], [], {}
    for i, (source, entry) in enumerate(zip(record["source_sentences"], record["labels"])):
        if entry["label"] != "EDIT":
            sources.append(source)
            labels.append(entry)
            continue
        target = entry["target"]
        kept = [] if i in undo else spans[i]
        text = target
        for c, d, _, new in reversed(kept):
            text = text[:c] + new + text[d:]
        sources.append(target)
        if kept:
            labels.append({"label": "EDIT", "stale_spans": [[c, d] for c, d, _, _ in kept], "target": text})
            changes[i] = [(old, new) for _, _, old, new in kept]
        else:
            labels.append({"label": "KEEP", "stale_spans": [], "target": target})
    targets = [entry["target"] for entry in labels if entry["label"] != "DROP"]
    out = {**record, "source_sentences": sources, "labels": labels, "evidence": [dict(p) for p in record["evidence"]],
           "target_sentences": targets, "target": " ".join(targets), "cascades": links}
    return out, changes


def substitute(record, changes):
    """(record with evidence stating the mutated strings, None) or (None, reason)."""
    links = record["cascades"]
    cascade_ids = {link["cascade"] for link in links}
    direct = {i: pairs for i, pairs in changes.items() if i not in cascade_ids}
    mapping = {old.lower(): new for pairs in direct.values() for old, new in pairs}
    for old in mapping:
        for i, source in enumerate(record["source_sentences"]):
            if has(old, source) and not any(o.lower() == old for o, _ in changes.get(i, [])):
                return None, "shared"
    evidence = record["evidence"]
    if mapping:
        pattern = re.compile(r"(?<!\w)(%s)(?!\w)" % "|".join(
            re.escape(o) for o in sorted(mapping, key=len, reverse=True)), re.IGNORECASE)
        swap = lambda m: same_case(mapping[m.group(1).lower()], m.group(1))
        evidence = [{"title": pattern.sub(swap, p["title"]), "text": pattern.sub(swap, p["text"])} for p in evidence]
    full = " ".join(f"{p['title']} {p['text']}" for p in evidence)
    for pairs in direct.values():
        for old, new in pairs:
            if not has(new, full):
                return None, "unsupported"
            if has(old, full):
                return None, "leftover"
    for link in links:
        news = [new for _, new in changes[link["direct"]]]
        for _, new in changes[link["cascade"]]:
            if new.lower() in PRONOUNS or any(has(new, n) for n in news):
                continue
            if has(new, full):
                return None, "cascade"
    return {**record, "evidence": evidence}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("--flip-only", action="store_true", help="keep the old evidence and run no evidence check")
    parser.add_argument("--rejects", help="write {id: reason} of the records left out here")
    parser.add_argument("--undo", help="JSON {id: [[direct, cascade], ...]} of links to undo")
    args = parser.parse_args()
    stats, rejects = Counter(), {}
    undo = json.loads(Path(args.undo).read_text()) if args.undo else {}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.data) as f, open(args.out, "w") as out:
        for line in f:
            record = json.loads(line)
            flipped, changes = flip(record, undo.get(record["id"], ()))
            reason = changes if flipped is None else None
            if flipped is not None and not args.flip_only:
                flipped, reason = substitute(flipped, changes)
            if reason:
                stats[f"rejected_{reason}"] += 1
                rejects[record["id"]] = reason
                continue
            stats["kept"] += 1
            stats["links_removed"] += len(record.get("cascades", [])) - len(flipped["cascades"])
            out.write(json.dumps(flipped, ensure_ascii=False) + "\n")
    if args.rejects:
        Path(args.rejects).write_text(json.dumps(rejects, indent=1))
    print(dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
