"""Step through a label audit sheet in the terminal and record verdicts.

Works on train_audit.tsv and flagged.tsv. Each row shows the source sentence
with its stale spans highlighted, the aligned target with inserted words
highlighted, the neighbouring sentences, and the evidence. Verdicts are saved
to the sheet after every row, and the review resumes at the first row without
a verdict.

Keys (then Enter):
    (empty)  the given label is right
    k e d    the right label is KEEP, EDIT or DROP
    s        the sentence is a bad split (not a real sentence)
    u        unsure
    n        add a note to this row
    v        show all evidence snippets, not only the cited ones
    b        back one row
    j N      jump to row N
    q        save and quit

After a verdict of EDIT or DROP on train_audit.tsv, a second prompt asks
whether the evidence supports the change: y, n, or Enter to skip.

Each reviewer works on their own copy, annotations/<name>/<sheet>, made from
the blank sheet on first use, so reviewers never overwrite each other.
--agreement compares every reviewer's copy of one sheet.

Usage:
    python review.py train_audit.tsv --annotator vedant --evidence train_audit_evidence.tsv
    python review.py flagged.tsv --annotator vedant --evidence flagged_evidence.tsv
    python review.py train_audit.tsv --annotator vedant --summary
    python review.py train_audit.tsv --agreement
"""

import argparse
import csv
import difflib
import json
import os
import re
import shutil
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

BOLD, DIM, RED, GREEN, CYAN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[36m", "\033[33m", "\033[0m")
LABEL_KEYS = {"k": "KEEP", "e": "EDIT", "d": "DROP", "s": "SPLIT", "u": "UNSURE"}
WORD = re.compile(r"\w+|[^\w\s]")
ANNOTATIONS = Path(__file__).resolve().parent / "annotations"
REVIEW_COLS = ("verdict", "supported", "notes")


def load_sheet(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return reader.fieldnames, list(reader)


def save_sheet(path, fields, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_evidence(sheet_rows, evidence_path, data_path):
    evidence = defaultdict(list)
    if evidence_path:
        with open(evidence_path, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                evidence[r["id"]].append((int(r["ref"]), r["title"], r["text"]))
    missing = {r["id"] for r in sheet_rows} - set(evidence)
    if data_path and missing:
        print(f"reading evidence for {len(missing)} articles from {data_path} ...")
        with open(data_path) as f:
            for line in f:
                if not missing:
                    break
                record = json.loads(line)
                if record["id"] in missing:
                    missing.discard(record["id"])
                    for k, e in enumerate(record["evidence"]):
                        evidence[record["id"]].append((k, e["title"], e["text"]))
    return evidence


def highlight_stale(source, stale):
    out, pos = [], 0
    for span in (s for s in stale.split(" | ") if s):
        i = source.find(span, pos)
        if i < 0:
            continue
        out += [source[pos:i], RED + BOLD + span + RESET]
        pos = i + len(span)
    return "".join(out) + source[pos:]


def highlight_inserted(source, target):
    a, b = WORD.findall(source.lower()), list(WORD.finditer(target))
    matcher = difflib.SequenceMatcher(a=a, b=[m.group().lower() for m in b], autojunk=False)
    marked = set()
    for op, _, _, j1, j2 in matcher.get_opcodes():
        if op in ("replace", "insert"):
            marked.update(range(j1, j2))
    out, pos = [], 0
    for k, m in enumerate(b):
        out.append(target[pos:m.start()])
        out.append(GREEN + BOLD + m.group() + RESET if k in marked else m.group())
        pos = m.end()
    return "".join(out) + target[pos:]


def wrap(text, indent, width):
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent,
                         break_long_words=False, break_on_hyphens=False)


def show(rows, i, label_col, evidence, show_all, width):
    r = rows[i]
    print("\033[2J\033[H", end="")
    done = sum(1 for x in rows if x.get("verdict"))
    print(f"{DIM}row {i + 1}/{len(rows)}   reviewed {done}{RESET}")
    print(f"{BOLD}{r['id']}  sentence {r['sentence']}{RESET}   given {YELLOW}{BOLD}{r[label_col]}{RESET}", end="")
    if "likely" in r:
        print(f"   model says {CYAN}{r['likely']}{RESET} ({r['p_likely']} vs {r['p_given']})", end="")
    if r.get("similarity"):
        print(f"   similarity {r['similarity']}", end="")
    print()

    same = [x for x in rows if x["id"] == r["id"]]
    k = next(n for n, x in enumerate(same) if x is r)
    if k > 0:
        print(DIM + wrap("before: " + same[k - 1]["source"], "  ", width) + RESET)
    print(f"\n{BOLD}source{RESET}")
    print(wrap(highlight_stale(r["source"], r.get("stale_tokens") or r.get("stale_spans", "")), "  ", width + 40))
    target = r.get("matched_target") or r.get("aligned_target", "")
    print(f"{BOLD}target{RESET}")
    print(wrap(highlight_inserted(r["source"], target), "  ", width + 40) if target else f"  {DIM}(none: no aligned target){RESET}")
    if k + 1 < len(same):
        print(DIM + wrap("after: " + same[k + 1]["source"], "  ", width) + RESET)

    cited = {int(c) for c in (r.get("target_evidence_refs") or "").split()}
    snippets = evidence.get(r["id"], [])
    shown = snippets if show_all or not cited else [s for s in snippets if s[0] in cited]
    label = "all evidence" if show_all or not cited else f"cited evidence ({len(shown)} of {len(snippets)}, v for all)"
    print(f"\n{BOLD}{label}{RESET}")
    if not snippets:
        print(f"  {DIM}(none loaded){RESET}")
    for ref, title, text in shown:
        mark = CYAN + "*" + RESET if ref in cited else " "
        print(wrap(f"{mark}({ref}) {title}: {text[:600]}", "  ", width))
    if r.get("verdict") or r.get("notes"):
        print(f"\n{DIM}saved: verdict={r.get('verdict', '')} supported={r.get('supported', '')} notes={r.get('notes', '')}{RESET}")


def summary(rows, label_col):
    judged = [r for r in rows if r.get("verdict") in ("KEEP", "EDIT", "DROP")]
    print(f"\n{len([r for r in rows if r.get('verdict')])} of {len(rows)} rows reviewed")
    if not judged:
        return
    wrong = [r for r in judged if r["verdict"] != r[label_col]]
    print(f"{len(wrong)} of {len(judged)} judged labels wrong ({len(wrong) / len(judged):.1%})")
    for given in ("KEEP", "EDIT", "DROP"):
        rows_g = [r for r in judged if r[label_col] == given]
        if rows_g:
            c = Counter(r["verdict"] for r in rows_g)
            print(f"  given {given}: {len(rows_g)} judged, " + ", ".join(f"{v} {n}" for v, n in c.most_common()))
    other = Counter(r["verdict"] for r in rows if r.get("verdict") in ("SPLIT", "UNSURE"))
    if other:
        print("  " + ", ".join(f"{v} {n}" for v, n in other.items()))
    supported = Counter(r["supported"] for r in rows if r.get("supported"))
    if supported:
        print(f"  evidence supports the change: yes {supported['y']}, no {supported['n']}")


def working_copy(sheet, annotator):
    path = ANNOTATIONS / annotator / Path(sheet).name
    if not path.exists():
        fields, rows = load_sheet(sheet)
        fields += [c for c in REVIEW_COLS if c not in fields]
        for r in rows:
            for c in REVIEW_COLS:
                r[c] = ""
        path.parent.mkdir(parents=True, exist_ok=True)
        save_sheet(str(path), fields, rows)
        print(f"created {path}")
    return str(path)


def kappa(a, b):
    """Cohen's kappa: agreement corrected for the agreement expected by chance."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return observed, (observed - expected) / (1 - expected) if expected < 1 else 1.0


def agreement(sheet):
    copies = sorted(ANNOTATIONS.glob(f"*/{Path(sheet).name}"))
    verdicts = {}
    for path in copies:
        _, rows = load_sheet(str(path))
        verdicts[path.parent.name] = {(r["id"], r["sentence"]): r["verdict"] for r in rows if r.get("verdict")}
        print(f"{path.parent.name}: {len(verdicts[path.parent.name])} rows reviewed")
    names = list(verdicts)
    if len(names) < 2:
        print("agreement needs at least two reviewers")
        return
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            shared = sorted(set(verdicts[x]) & set(verdicts[y]))
            if not shared:
                print(f"{x} vs {y}: no rows reviewed by both")
                continue
            observed, k = kappa([verdicts[x][s] for s in shared], [verdicts[y][s] for s in shared])
            print(f"{x} vs {y}: {len(shared)} shared rows, agree {observed:.1%}, Cohen's kappa {k:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sheet")
    parser.add_argument("--evidence", help="evidence TSV with id, ref, title, text")
    parser.add_argument("--data", help="collated jsonl to read evidence from")
    parser.add_argument("--summary", action="store_true", help="print the tally and exit")
    parser.add_argument("--annotator", help="your name; verdicts go to annotations/<name>/")
    parser.add_argument("--agreement", action="store_true", help="compare all reviewers of this sheet")
    args = parser.parse_args()

    if args.agreement:
        agreement(args.sheet)
        return
    if args.annotator:
        args.sheet = working_copy(args.sheet, args.annotator)
    fields, rows = load_sheet(args.sheet)
    label_col = "label" if "label" in fields else "given"
    for col in ("verdict", "supported", "notes"):
        if col not in fields:
            fields.append(col)
    if args.summary:
        summary(rows, label_col)
        return
    evidence = load_evidence(rows, args.evidence, args.data)
    width = min(shutil.get_terminal_size().columns, 120) - 4

    i = next((n for n, r in enumerate(rows) if not r.get("verdict")), len(rows) - 1)
    show_all = False
    while True:
        show(rows, i, label_col, evidence, show_all, width)
        try:
            cmd = input(f"\n{BOLD}[Enter]{RESET} right  {BOLD}k/e/d{RESET} label  {BOLD}s{RESET} split  "
                        f"{BOLD}u{RESET} unsure  {BOLD}n{RESET} note  {BOLD}v{RESET} evidence  "
                        f"{BOLD}b{RESET} back  {BOLD}j N{RESET} jump  {BOLD}q{RESET} quit > ").strip().lower()
        except EOFError:
            cmd = "q"
        r = rows[i]
        if cmd == "q":
            break
        if cmd == "v":
            show_all = not show_all
            continue
        if cmd == "b":
            i, show_all = max(i - 1, 0), False
            continue
        if cmd.startswith("j"):
            try:
                i, show_all = min(max(int(cmd[1:]) - 1, 0), len(rows) - 1), False
            except ValueError:
                pass
            continue
        if cmd == "n":
            r["notes"] = input("note > ").strip()
            save_sheet(args.sheet, fields, rows)
            continue
        if cmd == "":
            r["verdict"] = r[label_col]
        elif cmd in LABEL_KEYS:
            r["verdict"] = LABEL_KEYS[cmd]
        else:
            continue
        if "target_evidence_refs" in fields and r["verdict"] in ("EDIT", "DROP"):
            answer = input("evidence supports the change? y / n / Enter to skip > ").strip().lower()
            r["supported"] = answer if answer in ("y", "n") else ""
        save_sheet(args.sheet, fields, rows)
        if i == len(rows) - 1:
            break
        i, show_all = i + 1, False
    save_sheet(args.sheet, fields, rows)
    summary(rows, label_col)


if __name__ == "__main__":
    sys.exit(main())
