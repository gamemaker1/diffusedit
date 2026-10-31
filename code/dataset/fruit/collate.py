"""Builds the collated DiffusEdit splits from FRUIT TFRecord files.

Reconstruction of the original collation script, checked byte for byte
against collated/test.jsonl. Each FRUIT instance becomes one record with
sentence labels, mentions, and evidence cut to the token budget.

Pipeline per instance:
  1. Parse the TFRecord (fruit/load.py). Target sentences are the copied
     source sentences plus the spaCy sentences of each generated span.
  2. Drop the instance if the article exceeds 400 LLaDA tokens.
  3. Align source to target sentences greedily by token-multiset F1
     (floor 0.35, ties to the nearest position). Unaligned source
     sentences are DROP, aligned ones KEEP (F1 >= 0.95 and no changed
     name or number) or EDIT, and unaligned target sentences are ADD.
  4. Drop the instance if no sentence is EDIT or DROP (append-only).
  5. Extract mentions, classify changes, and cut evidence to 600 tokens.

Requirements: spacy 3.8 with en_core_web_sm 3.8.0, tokenizers, and the
LLaDA-8B-Instruct tokenizer.json.

Usage:
  python collate.py shard --input X.tfrecords --split train --out part.jsonl
  python collate.py merge --parts DIR --outdir DIR [--val-size 2000]
  python collate.py relabel --input train.jsonl --out train.relabelled.jsonl
"""

import argparse
import difflib
import glob
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "fruit"))
import load  # noqa: E402

ARTICLE_CAP = 400
EVIDENCE_CAP = 600
ARTICLE_OVERHEAD = 5
EVIDENCE_OVERHEAD = 3
ALIGN_FLOOR = 0.35
KEEP_SIM = 0.95
VAL_SEED = 20260927

MENTION_LABELS = {
    "CARDINAL", "DATE", "FAC", "GPE", "LOC", "MONEY", "NORP", "ORDINAL",
    "ORG", "PERCENT", "PERSON", "QUANTITY", "TIME",
}
MENTION_STRIP = " .,;:'\"()!?\t\n"
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
TOKEN_RE = re.compile(r"\w+|[^\w\s]")

_nlp = None
_tok = None


def nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def tokenizer():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        path = os.environ.get("LLADA_TOKENIZER")
        if not path:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("GSAI-ML/LLaDA-8B-Instruct",
                                   "tokenizer.json")
        _tok = Tokenizer.from_file(path)
    return _tok


def n_tokens(text):
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


# ---------------------------------------------------------------------------
# Mentions
# ---------------------------------------------------------------------------


def _norm_mention(text):
    return re.sub(r"\s+", " ", text.lower()).strip(MENTION_STRIP)


def mentions_in(doc, start, end, text):
    """Mentions of doc entities inside [start, end) plus digit strings."""
    out = {_norm_mention(e.text) for e in doc.ents
           if e.start_char >= start and e.end_char <= end
           and e.label_ in MENTION_LABELS}
    out |= set(NUMBER_RE.findall(text))
    out.discard("")
    return sorted(out)


def article_mentions(sentences):
    doc = nlp()(" ".join(sentences))
    out, pos = [], 0
    for s in sentences:
        out.append(mentions_in(doc, pos, pos + len(s), s))
        pos += len(s) + 1
    return out


# ---------------------------------------------------------------------------
# Alignment and labels
# ---------------------------------------------------------------------------


def words(text):
    return TOKEN_RE.findall(text.lower())


def similarity(a, b):
    if not a or not b:
        return 0.0
    overlap = sum((Counter(a) & Counter(b)).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(a), overlap / len(b)
    return 2 * p * r / (p + r)


def align(source, target):
    """Greedy one-to-one alignment by similarity, ties to nearest index."""
    a = [words(s) for s in source]
    b = [words(t) for t in target]
    sim = [[similarity(x, y) for y in b] for x in a]
    cands = sorted(((sim[i][j], i, j) for i in range(len(a))
                    for j in range(len(b))),
                   key=lambda c: (-c[0], abs(c[1] - c[2]), c[1]))
    pairs, used = {}, set()
    for s, i, j in cands:
        if s < ALIGN_FLOOR:
            break
        if i in pairs or j in used:
            continue
        pairs[i] = j
        used.add(j)
    return pairs, sim


def edit_fields(source, target):
    """Stale character spans in source and the removed/inserted words."""
    a = list(TOKEN_RE.finditer(source))
    b = list(TOKEN_RE.finditer(target))
    matcher = difflib.SequenceMatcher(
        a=[m.group().lower() for m in a], b=[m.group().lower() for m in b],
        autojunk=False)
    stale, removed, inserted, anchor = set(), [], [], False
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        if op in ("replace", "delete"):
            stale.update(range(i1, i2))
            removed += [m.group() for m in a[i1:i2]]
        if op in ("replace", "insert"):
            inserted += [m.group() for m in b[j1:j2]]
        if op == "insert":
            anchor = True
            if a:
                stale.add(i1 - 1 if i1 > 0 else 0)
    spans = []
    for k in sorted(stale):
        if spans and spans[-1][2] == k - 1:
            spans[-1][1], spans[-1][2] = a[k].end(), k
        else:
            spans.append([a[k].start(), a[k].end(), k])
    return {
        "stale_spans": [[s, e] for s, e, _ in spans],
        "removed": " ".join(removed),
        "inserted": " ".join(inserted),
        "insert_anchor": anchor,
    }


def content_changed(source, target):
    """True if the word diff touches a word with a capital letter or digit.

    A one-word change such as a new club name keeps F1 above KEEP_SIM in a
    long sentence, so similarity alone labels it KEEP.
    """
    a = [m.group() for m in TOKEN_RE.finditer(source)]
    b = [m.group() for m in TOKEN_RE.finditer(target)]
    matcher = difflib.SequenceMatcher(
        a=[w.lower() for w in a], b=[w.lower() for w in b], autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        for w in a[i1:i2] + b[j1:j2]:
            if any(c.isupper() or c.isdigit() for c in w):
                return True
    return False


def label_sentences(source, target):
    pairs, sim = align(source, [t["text"] for t in target])
    labels = []
    for i, s in enumerate(source):
        j = pairs.get(i)
        if j is None:
            labels.append({"label": "DROP", "target_idx": None,
                           "similarity": 0.0})
            continue
        score = round(sim[i][j], 4)
        if score >= KEEP_SIM and not content_changed(s, target[j]["text"]):
            labels.append({"label": "KEEP", "target_idx": j,
                           "similarity": score})
        else:
            labels.append({"label": "EDIT", "target_idx": j,
                           "similarity": score,
                           **edit_fields(s, target[j]["text"])})
    aligned = set(pairs.values())
    additions = [j for j in range(len(target)) if j not in aligned]
    return labels, additions


# ---------------------------------------------------------------------------
# Changes
# ---------------------------------------------------------------------------


def changed_mentions(source, source_m, target):
    sm = set(source_m)
    if target is None:
        return sm, sm
    tm = set(target["mentions"])
    changed = ({m for m in sm - tm if m not in target["text"].lower()} |
               {m for m in tm - sm if m not in source.lower()})
    return changed, sm | tm


def classify_changes(record):
    evidence_m = {m for e in record["evidence"] for m in e["mentions"]}
    entries = []
    for i, lab in enumerate(record["labels"]):
        if lab["label"] == "KEEP":
            continue
        tgt = (None if lab["target_idx"] is None
               else record["target_sentences"][lab["target_idx"]])
        changed, all_m = changed_mentions(
            record["source_sentences"][i], record["source_mentions"][i], tgt)
        entries.append((i, changed, all_m, bool(changed & evidence_m)))
    direct_m = set()
    for _, _, all_m, direct in entries:
        if direct:
            direct_m |= all_m
    changes = []
    for i, changed, all_m, direct in entries:
        entry = {"idx": i, "kind": "direct" if direct else "candidate",
                 "changed_mentions": sorted(changed)}
        if not direct:
            entry["shares_mention_with_direct"] = bool(all_m & direct_m)
        changes.append(entry)
    return changes


# ---------------------------------------------------------------------------
# Evidence budget
# ---------------------------------------------------------------------------


def evidence_tokens(passages):
    rendered = "\n".join(f"[{i}] {p['title']}: {p['text']}"
                         for i, p in enumerate(passages))
    return n_tokens(rendered) + EVIDENCE_OVERHEAD


def cut_evidence(evidence, source_mentions):
    """Fills passages by mention overlap with the article, cutting at 600."""
    src = {m for ms in source_mentions for m in ms}
    full = [e["full_text"] for e in evidence]
    out = [{"title": e["title"], "text": "", "truncated": True}
           for e in evidence]
    order = sorted(range(len(evidence)),
                   key=lambda i: (-len(set(evidence[i]["mentions"]) & src), i))
    for i in order:
        out[i]["text"], out[i]["truncated"] = full[i], False
        if evidence_tokens(out) <= EVIDENCE_CAP:
            continue
        out[i]["truncated"] = True
        ids = tokenizer().encode(full[i], add_special_tokens=False).ids
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            out[i]["text"] = tokenizer().decode(ids[:mid]).strip()
            if evidence_tokens(out) <= EVIDENCE_CAP:
                lo = mid
            else:
                hi = mid - 1
        out[i]["text"] = tokenizer().decode(ids[:lo]).strip()
        break
    result = []
    for e, o, f in zip(evidence, out, full):
        item = {"title": e["title"], "text": o["text"]}
        if o["truncated"]:
            item["full_text"] = f
        item["truncated"] = o["truncated"]
        item["mentions"] = e["mentions"]
        result.append(item)
    return result, evidence_tokens(out)


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------


def target_sentences(inst, source_m):
    out = []
    for op in inst["target_ops"]:
        if op["op"] == "copy":
            i = op["source_idx"]
            out.append({"text": inst["source_sentences"][i], "origin": "copy",
                        "source_idx": i, "evidence_refs": [],
                        "mentions": source_m[i]})
            continue
        doc = nlp()(op["text"])
        for sent in doc.sents:
            text = sent.text.strip()
            if not text:
                continue
            out.append({"text": text, "origin": "generate",
                        "source_idx": None,
                        "evidence_refs": op["evidence_refs"],
                        "mentions": mentions_in(doc, sent.start_char,
                                                sent.end_char, sent.text)})
    return out


def build(features, source_id):
    """Returns (record or None, reason)."""
    inst = load.parse_instance(features)
    source = inst["source_sentences"]
    article_tokens = n_tokens(" ".join(source)) + ARTICLE_OVERHEAD
    if article_tokens > ARTICLE_CAP:
        return None, "article_over_budget"
    source_m = article_mentions(source)
    target = target_sentences(inst, source_m)
    labels, additions = label_sentences(source, target)
    if all(lab["label"] == "KEEP" for lab in labels):
        return None, "append_only"
    evidence = []
    for e in inst["evidence"]:
        title = e["title"].replace("_", " ")
        doc = nlp()(f"{title}: {e['content']}")
        evidence.append({"title": title, "full_text": e["content"],
                         "mentions": mentions_in(doc, 0, len(doc.text),
                                                 doc.text)})
    record = {
        "source_id": source_id,
        "page_id": features["id"],
        "source_sentences": source,
        "source_mentions": source_m,
        "target_sentences": target,
        "labels": labels,
        "additions": additions,
    }
    record["evidence"] = evidence
    record["changes"] = classify_changes(record)
    counts = Counter(lab["label"] for lab in labels)
    record["counts"] = {"KEEP": counts["KEEP"], "EDIT": counts["EDIT"],
                        "DROP": counts["DROP"], "ADD": len(additions)}
    record["article_tokens"] = article_tokens
    record["evidence"], record["evidence_tokens"] = cut_evidence(
        evidence, source_m)
    order = ["source_id", "page_id", "source_sentences", "source_mentions",
             "target_sentences", "labels", "additions", "changes", "counts",
             "article_tokens", "evidence", "evidence_tokens"]
    return {k: record[k] for k in order}, "kept"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_shard(args):
    """Processes one TFRecord file into kept records plus a tally."""
    name = os.path.basename(args.input).split(".")[0]
    tally = Counter()
    with open(args.out, "w", encoding="utf-8") as f:
        for index, feats in enumerate(load.read_tfrecords(args.input)):
            record, reason = build(feats, f"{args.split}/{name}/{index}")
            tally[reason] += 1
            if record is not None:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(args.out + ".tally.json", "w") as f:
        json.dump(dict(tally), f)
    print(args.input, dict(tally), flush=True)


def finalize(records, split, quality):
    for n, r in enumerate(records):
        r["id"] = f"{split}-{n:06d}"
        r["split"] = split
        r["labels_quality"] = quality
    return records


def read_parts(paths):
    records, tally = [], Counter()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            records += [json.loads(line) for line in f]
        with open(p + ".tally.json") as f:
            tally.update(json.load(f))
    return records, tally


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_merge(args):
    """Joins shard parts: shards 0-8 train, shard 9 dev -> val + train."""
    parts = sorted(glob.glob(os.path.join(args.parts, "*.jsonl")))
    dev_parts = [p for p in parts if "-00009-of-00010" in p]
    train_parts = [p for p in parts if p not in dev_parts]
    train, train_tally = read_parts(train_parts)
    dev, dev_tally = read_parts(dev_parts)
    val_idx = set(random.Random(VAL_SEED).sample(range(len(dev)),
                                                 args.val_size))
    val = [r for i, r in enumerate(dev) if i in val_idx]
    train += [r for i, r in enumerate(dev) if i not in val_idx]
    os.makedirs(args.outdir, exist_ok=True)
    write_jsonl(os.path.join(args.outdir, "train.jsonl"),
                finalize(train, "train", "silver"))
    write_jsonl(os.path.join(args.outdir, "val.jsonl"),
                finalize(val, "val", "silver"))
    with open(os.path.join(args.outdir, "rebuild_tallies.json"), "w") as f:
        json.dump({"train": dict(train_tally), "dev": dict(dev_tally)}, f,
                  indent=2)
    print("train", len(train), dict(train_tally))
    print("dev", len(dev), dict(dev_tally), "val", len(val))


def cmd_test(args):
    """Rebuilds the gold test split for comparison with collated/test.jsonl."""
    records, tally = [], Counter()
    for index, feats in enumerate(load.read_tfrecords(args.input)):
        record, reason = build(feats, f"test/gold_test/{index}")
        tally[reason] += 1
        if record is not None:
            records.append(record)
    write_jsonl(args.out, finalize(records, "test", "gold"))
    print(dict(tally))


def cmd_relabel(args):
    """Applies the current KEEP rule to existing records, keeping their ids.

    Only KEEP labels can change, to EDIT. Records dropped earlier as
    append-only stay out, so a full rebuild would keep a few more articles.
    """
    moved = Counter()
    with open(args.input, encoding="utf-8") as fin, \
            open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            changed = False
            for i, lab in enumerate(r["labels"]):
                if lab["label"] != "KEEP":
                    continue
                src = r["source_sentences"][i]
                tgt = r["target_sentences"][lab["target_idx"]]["text"]
                if content_changed(src, tgt):
                    r["labels"][i] = {"label": "EDIT",
                                      "target_idx": lab["target_idx"],
                                      "similarity": lab["similarity"],
                                      **edit_fields(src, tgt)}
                    moved["KEEP->EDIT"] += 1
                    changed = True
            if changed:
                r["changes"] = classify_changes(r)
                counts = Counter(lab["label"] for lab in r["labels"])
                r["counts"].update({k: counts[k] for k in ("KEEP", "EDIT", "DROP")})
                moved["articles"] += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(moved))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--input", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--parts", required=True)
    m.add_argument("--outdir", required=True)
    m.add_argument("--val-size", type=int, default=2000)
    t = sub.add_parser("test")
    t.add_argument("--input", required=True)
    t.add_argument("--out", required=True)
    r = sub.add_parser("relabel")
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    {"shard": cmd_shard, "merge": cmd_merge, "test": cmd_test,
     "relabel": cmd_relabel}[args.cmd](args)


if __name__ == "__main__":
    main()
