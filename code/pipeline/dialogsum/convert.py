"""Turns DialogSum into DiffusEdit records, so the existing extract and train
code can read it.

DialogSum has dialogues with human summaries and nothing else: no outdated
text, no evidence snippets, no KEEP/EDIT/DROP labels. The task is set up the
way Detect-Remask-Repair (Zou et al., 2026) sets it up:

  1. The dialogue is cut after its first --frac of turns (the early context).
  2. A language model writes a draft summary of the early context only. That
     draft is the stale text. This script does not run the model: `prompts`
     writes what to give it, and `build` reads what it wrote.
  3. The full dialogue is the evidence. The human summary of the full
     dialogue is the updated text.
  4. The draft's sentences are labelled against the human summary with
     collate.label_sentences, the function the FRUIT labels came from.

Sentences of the human summary with no partner in the draft are additions.
The pipeline neither labels nor generates additions, so a draft that only
lacks information ends up all KEEP and is dropped as "unchanged", the same
way collate.py drops append-only FRUIT articles. The report at the end of
`build` says how often that happens.

Labels come from the first reference summary. The test split has three; all
of them are kept under "references" for scoring.

Usage (from the pipeline folder):
    python -m dialogsum.convert prompts --data ../../datasets/dialogsum/official \\
        --split test --out work/test.prompts.jsonl
    # run any model over work/test.prompts.jsonl and write one line per id:
    #   {"id": "dialogsum-test-test_0", "draft": "..."}
    python -m dialogsum.convert build --data ../../datasets/dialogsum/official \\
        --split test --drafts work/test.drafts.jsonl --out work/test.jsonl
"""

import argparse
import json
import math
import re
import sys
import types
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
SPLITS = {"train": "dialogsum.train.jsonl", "dev": "dialogsum.dev.jsonl", "test": "dialogsum.test.jsonl"}
TURN = re.compile(r"^#Person\d+#:")
ABBREVIATION = re.compile(r"\b(?:mr|mrs|ms|dr|st|jr|sr|vs|prof)\.$", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(]?[A-Z#])")
PROMPT = ("Summarize the dialogue below in one to three short sentences. "
          "Call the speakers #Person1# and #Person2#.\n\n{dialogue}\n\nSummary:")
# 600 evidence tokens is roughly 450 words of dialogue
LONG_WORDS = 450


def labeller():
    """collate.py, imported without its FRUIT loader (not in this repo)."""
    sys.modules.setdefault("load", types.ModuleType("load"))
    sys.path.insert(0, str(CODE / "datasets"))
    import collate
    return collate


def turns(dialogue):
    out = []
    for line in dialogue.split("\n"):
        line = line.strip()
        if not line:
            continue
        if TURN.match(line) or not out:
            out.append(line)
        else:
            out[-1] += " " + line
    return out


def early_context(dialogue, frac):
    t = turns(dialogue)
    return "\n".join(t[:max(1, math.ceil(len(t) * frac))])


def sentences(text):
    out = []
    for part in SENTENCE_END.split(text.strip()):
        if out and ABBREVIATION.search(out[-1]):
            out[-1] += " " + part
        else:
            out.append(part)
    return [s.strip() for s in out if s.strip()]


def read_split(data, split):
    """Rows with the id, the dialogue and its reference summaries. Drops
    duplicate dialogues, one-turn dialogues and empty summaries."""
    seen, rows, dropped = set(), [], {"duplicate": 0, "one_turn": 0, "no_summary": 0}
    with open(Path(data) / SPLITS[split], encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            refs = [raw[k].strip() for k in ("summary", "summary1", "summary2", "summary3")
                    if raw.get(k) and raw[k].strip()]
            key = raw["dialogue"].strip()
            if key in seen:
                dropped["duplicate"] += 1
            elif len(turns(key)) < 2:
                dropped["one_turn"] += 1
            elif not refs:
                dropped["no_summary"] += 1
            else:
                seen.add(key)
                rows.append({"id": f"dialogsum-{split}-{raw['fname']}", "dialogue": key, "references": refs})
    return rows, dropped


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def cmd_prompts(args):
    rows, dropped = read_split(args.data, args.split)
    rows = rows[:args.limit]
    early = [early_context(r["dialogue"], args.frac) for r in rows]
    write_jsonl(args.out, ({"id": r["id"], "early_dialogue": e, "prompt": PROMPT.format(dialogue=e)}
                           for r, e in zip(rows, early)))
    print(f"{len(rows)} prompts written to {args.out} (dropped before this: {dropped})")


def cmd_build(args):
    collate = labeller()
    rows, dropped = read_split(args.data, args.split)
    drafts = {}
    with open(args.drafts, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                drafts[entry["id"]] = entry["draft"].strip()

    records, skipped = [], {"no_draft": 0, "unchanged": 0}
    counts = {"KEEP": 0, "EDIT": 0, "DROP": 0}
    additions = long_dialogues = 0
    for row in rows[:args.limit]:
        draft = drafts.get(row["id"], "")
        if not draft:
            skipped["no_draft"] += 1
            continue
        source = sentences(draft)
        target = [{"text": t} for t in sentences(row["references"][0])]
        labels, added = collate.label_sentences(source, target)
        if not args.keep_unchanged and all(lab["label"] == "KEEP" for lab in labels):
            skipped["unchanged"] += 1
            continue
        for lab in labels:
            counts[lab["label"]] += 1
        additions += len(added)
        long_dialogues += len(row["dialogue"].split()) > LONG_WORDS
        records.append({
            "id": row["id"],
            "source_sentences": source,
            "target_sentences": target,
            "labels": labels,
            "additions": added,
            "evidence": [{"title": "Dialogue", "text": row["dialogue"]}],
            "references": row["references"],
            "early_frac": args.frac,
        })
    write_jsonl(args.out, records)

    n = max(len(records), 1)
    total = max(sum(counts.values()), 1)
    print(f"{len(records)} records written to {args.out}")
    print(f"dropped while reading: {dropped}")
    print(f"skipped: {skipped}")
    print("draft sentence labels: " + "  ".join(f"{k} {v} ({v / total:.0%})" for k, v in counts.items()))
    print(f"additions (human sentences with no draft partner): {additions}, {additions / n:.2f} per record")
    print(f"dialogues over ~{LONG_WORDS} words, whose evidence the 600-token cut will shorten: {long_dialogues}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("prompts", cmd_prompts), ("build", cmd_build)):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True, help="folder holding the official dialogsum .jsonl files")
        p.add_argument("--split", choices=list(SPLITS), required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--frac", type=float, default=0.5, help="share of turns the draft gets to see")
        p.add_argument("--limit", type=int, help="first N dialogues only")
        if name == "build":
            p.add_argument("--drafts", required=True, help="jsonl of {id, draft}")
            p.add_argument("--keep-unchanged", action="store_true", help="keep records whose draft sentences are all KEEP")
        p.set_defaults(func=fn)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
