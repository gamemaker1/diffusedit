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

The gender kind gave the subject a first name of the swapped gender in the
mutated sentences. That swap is undone, so the target keeps the real name.

A record is left out when a check fails:

    insertion   a stale span whose old gold text is empty, which has no
                source token to mark
    conflict    one old gold string maps to two mutated strings
    shared      an old gold string of a direct edit occurs in a source
                sentence that does not change it, so the replacement would
                also rewrite the evidence of that sentence
    unsupported a mutated string of a direct edit is not in the new evidence
    leftover    an old gold string of a direct edit is still in the evidence
    cascade     a mutated string of a cascade sentence that its direct edit
                does not contain is in the evidence (pronouns excepted, which
                the evidence uses for other people)
    ambiguous   a gender link whose direct edit turns "actress" into "actor",
                which is also used for women, so the new pronouns do not
                follow from it

A link whose direct sentence loses all its spans to the name rule is dropped
with that rule's record count.

Usage:
    python reverse.py split/train.jsonl reversed/train.jsonl
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


def reverse(record):
    """(reversed record, None) or (None, reason)."""
    cascades = {link["cascade"]: link for link in record.get("cascades", [])}
    gender = any(link.get("kind") == "gender" for link in cascades.values())
    sources, labels, changes = [], [], {}
    for i, (source, entry) in enumerate(zip(record["source_sentences"], record["labels"])):
        if entry["label"] != "EDIT":
            sources.append(source)
            labels.append(entry)
            continue
        target = entry["target"]
        kept = []
        for a, b, c, d in edits(source, target, entry["stale_spans"]):
            old, new = target[c:d], source[a:b]
            if not old.strip():
                return None, "insertion"
            if gender and new in NAMES and old[:1].isupper() and " " not in old and old.lower() not in PRONOUNS:
                continue
            kept.append((c, d, old, new))
        text = target
        for c, d, _, new in reversed(kept):
            text = text[:c] + new + text[d:]
        sources.append(target)
        if kept:
            labels.append({"label": "EDIT", "stale_spans": [[c, d] for c, d, _, _ in kept], "target": text})
            changes[i] = [(old, new) for _, _, old, new in kept]
        else:
            labels.append({"label": "KEEP", "stale_spans": [], "target": target})

    links = [link for link in record.get("cascades", []) if link["direct"] in changes and link["cascade"] in changes]
    cascade_ids = {link["cascade"] for link in links}
    direct = {i: pairs for i, pairs in changes.items() if i not in cascade_ids}

    mapping = {}
    for pairs in direct.values():
        for old, new in pairs:
            if mapping.setdefault(old.lower(), new).lower() != new.lower():
                return None, "conflict"
    for old in mapping:
        for i, source in enumerate(sources):
            if has(old, source) and not any(o.lower() == old for o, _ in changes.get(i, [])):
                return None, "shared"

    evidence = []
    if mapping:
        pattern = re.compile(r"(?<!\w)(%s)(?!\w)" % "|".join(
            re.escape(o) for o in sorted(mapping, key=len, reverse=True)), re.IGNORECASE)
        swap = lambda m: same_case(mapping[m.group(1).lower()], m.group(1))
        evidence = [{"title": pattern.sub(swap, p["title"]), "text": pattern.sub(swap, p["text"])}
                    for p in record["evidence"]]
    else:
        evidence = [dict(p) for p in record["evidence"]]
    full = " ".join(f"{p['title']} {p['text']}" for p in evidence)
    for pairs in direct.values():
        for old, new in pairs:
            if not has(new, full):
                return None, "unsupported"
            if has(old, full):
                return None, "leftover"
    for link in links:
        news = [new for _, new in changes[link["direct"]]]
        if link.get("kind") == "gender" and any(n.lower() in NEUTRAL for n in news):
            return None, "ambiguous"
        for _, new in changes[link["cascade"]]:
            if new.lower() in PRONOUNS or any(has(new, n) for n in news):
                continue
            if has(new, full):
                return None, "cascade"

    targets = [entry["target"] for entry in labels if entry["label"] != "DROP"]
    out = {**record, "source_sentences": sources, "labels": labels, "evidence": evidence,
           "target_sentences": targets, "target": " ".join(targets), "cascades": links}
    return out, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("out")
    args = parser.parse_args()
    stats = Counter()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.data) as f, open(args.out, "w") as out:
        for line in f:
            record = json.loads(line)
            reversed_record, reason = reverse(record)
            if reason:
                stats[f"rejected_{reason}"] += 1
                continue
            stats["kept"] += 1
            stats["links_dropped"] += len(record.get("cascades", [])) - len(reversed_record["cascades"])
            out.write(json.dumps(reversed_record, ensure_ascii=False) + "\n")
    print(dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
