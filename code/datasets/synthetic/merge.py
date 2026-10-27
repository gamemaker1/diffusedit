"""Replace the evidence of rebuilt records with written passages.

rebuild.py leaves evidence that copies the gold sentence. Writers (language
model agents following prompt.md) produce, per article, passages
that read as excerpts of other documents. This script joins them to the
records and keeps an article only if its passages pass every check:

    each stale span's gold text appears in a passage about that sentence
    no passage holds the wrong text of a stale span
    every DROP sentence has a passage about it
    no passage copies seven consecutive words of a sentence

Usage:
    python merge.py rebuilt.jsonl passages/ train.jsonl
"""

import argparse
import glob
import json
import random
import re
from collections import Counter


def words(text):
    return re.findall(r"\w+", text.lower())


def shingles(text, n=7):
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def gold_spans(source, target, spans):
    """The gold text of every stale span: the target text between the unchanged neighbours."""
    out, pos_s, pos_t = [], 0, 0
    for k, (a, b) in enumerate(spans):
        pos_t += a - pos_s
        after = source[b:spans[k + 1][0]] if k + 1 < len(spans) else source[b:]
        end = len(target) - len(after) if k + 1 == len(spans) else target.index(after, pos_t) if after else pos_t
        out.append((source[a:b], target[pos_t:end]))
        pos_s, pos_t = b, end
    return out


def check(record, passages):
    """None if the passages are valid for the record, else the reason."""
    if not passages or any(not isinstance(p, dict) or not p.get("title") or not p.get("text")
                           or not isinstance(p.get("about"), list) for p in passages):
        return "malformed"
    article = " ".join(record["source_sentences"]).lower() + " " + record["target"].lower()
    copied = set().union(*(shingles(s) for s in record["source_sentences"] + record["target_sentences"]))
    # a gold span of seven or more words has to appear in a passage, so it does not count as copying
    for source, entry in zip(record["source_sentences"], record["labels"]):
        if entry["label"] == "EDIT":
            for _, right in gold_spans(source, entry["target"], entry["stale_spans"]):
                copied -= shingles(right)
    if any(shingles(p["text"]) & copied for p in passages):
        return "copied"
    for j, (source, entry) in enumerate(zip(record["source_sentences"], record["labels"])):
        about = [p["text"].lower() for p in passages if j in p["about"]]
        if entry["label"] == "DROP" and not about:
            return "drop_without_passage"
        if entry["label"] != "EDIT":
            continue
        for wrong, right in gold_spans(source, entry["target"], entry["stale_spans"]):
            if not any(right.lower() in text for text in about):
                return "gold_absent"
            pattern = rf"(?<!\w){re.escape(wrong.lower())}(?!\w)"
            if article.count(wrong.lower()) == 1 and any(re.search(pattern, p["text"].lower()) for p in passages):
                return "wrong_present"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("passages", help="directory of chunk_*.jsonl files with {id, evidence}")
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rejects", help="write rejected ids and reasons here")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    written = {}
    stats = Counter()
    for path in sorted(glob.glob(f"{args.passages}/chunk_*.jsonl")):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    written[item["id"]] = item["evidence"]
                except (ValueError, KeyError, TypeError):
                    stats["unparseable_lines"] += 1

    rejects = {}
    with open(args.data) as f, open(args.out, "w") as out:
        for line in f:
            record = json.loads(line)
            if record["id"] not in written:
                stats["not_written"] += 1
                continue
            passages = written[record["id"]]
            reason = check(record, passages)
            if reason:
                stats[f"rejected_{reason}"] += 1
                rejects[record["id"]] = reason
                continue
            rng.shuffle(passages)
            record["evidence"] = [{"title": p["title"], "text": " ".join(p["text"].split())} for p in passages]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats["passages"] += len(passages)
    if args.rejects:
        with open(args.rejects, "w") as f:
            json.dump(rejects, f, indent=1)
    print(dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
