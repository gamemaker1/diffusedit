"""Makes stale-summary training data from DialogSum by corrupting the human
summaries, the way Detect-Remask-Repair (Zou et al., 2026) trains its detector.

No draft-writing model is needed. For each dialogue:

  1. Pick a noise level t between --min-noise and --max-noise.
  2. Mask a share t of the human summary's tokens, in short runs. Also mask a
     share t of the dialogue (the evidence). The refiller therefore cannot
     copy the answer back out of the dialogue, which is what makes its fill
     fluent but sometimes unsupported.
  3. Let frozen LLaDA fill the masked summary tokens (the infilling sampler).
     The masked dialogue tokens stay masked.
  4. Compare with the original summary. A token the refill changed is stale,
     every other token is not. The record keeps the ORIGINAL dialogue as
     evidence, so "stale" means "not supported by the dialogue".

DRR leaves some positions masked in the detector's input. Our token head reads
fully visible text, so every masked summary token is refilled here. That is
our adaptation, not the paper's.

The corrupted summary has the same tokens, in the same sentences, as the human
one. Sentence j of the record therefore pairs with sentence j of the human
summary (target_idx = j). A sentence with a changed token is EDIT, any other is
KEEP. With --drop-prob above 0, an unrelated sentence from another dialogue's
summary is inserted and labelled DROP; it is trivially off-topic, so treat DROP
results from it as a smoke test only.

Output records carry the fields convert.py's do (except early_frac), and the classifier,
staleness and infilling code read them unchanged. Reruns skip ids already in
--out.

Usage (from the pipeline folder, on a GPU):
    python -m dialogsum.corrupt --data ../../datasets/dialogsum/official \\
        --split train --out work/train.corrupt.jsonl --limit 2000
"""

import argparse
import json
import random
import re
from pathlib import Path

from dialogsum.convert import read_split, sentences

TOKEN = re.compile(r"\w+|[^\w\s]")
RUNS = (1, 1, 2, 3)


def corrupt_id(row):
    return row["id"].replace("dialogsum-", "dialogsum-corrupt-", 1)


def pick_positions(n, share, rng):
    """About `share` of n positions, chosen as short runs."""
    want = min(n, max(1, round(share * n)))
    chosen = set()
    while len(chosen) < want:
        start = rng.randrange(n)
        chosen.update(range(start, min(n, start + rng.choice(RUNS))))
    return sorted(chosen)


def sentence_tokens(text, offsets):
    """(sentence, first token, last token + 1) for each sentence of `text`."""
    out, pos = [], 0
    for sentence in sentences(text):
        start = text.find(sentence, pos)
        if start < 0:
            return None
        end = start + len(sentence)
        pos = end
        covered = [t for t, (a, b) in enumerate(offsets) if a < end and b > start]
        if not covered:
            return None
        out.append((sentence, covered[0], covered[-1] + 1))
    return out


def decode_spans(tok, ids):
    """Text of `ids` and each token's character range in it, or None if the
    decoded text of a prefix is not a prefix of the next (broken bytes)."""
    text, spans = "", []
    for i in range(len(ids)):
        new = tok.decode(ids[:i + 1], clean_up_tokenization_spaces=False)
        if not new.startswith(text):
            return None
        spans.append((len(text), len(new)))
        text = new
    return text, spans


def same_text_unchanged(tok, corrupt, gold):
    """Per token: did the refill change it? A run of different token ids that
    decodes to the same text as the original run (the model split a word into
    other pieces) is not a change."""
    changed = [a != b for a, b in zip(corrupt, gold)]
    i = 0
    while i < len(changed):
        if not changed[i]:
            i += 1
            continue
        j = i
        while j < len(changed) and changed[j]:
            j += 1
        if tok.decode(corrupt[i:j], clean_up_tokenization_spaces=False) == tok.decode(gold[i:j], clean_up_tokenization_spaces=False):
            changed[i:j] = [False] * (j - i)
        i = j
    return changed


def stale_word_spans(text, token_spans, changed):
    """Character spans of the whole words that a changed token touches,
    adjacent words joined, as collate.edit_fields gives them."""
    words = list(TOKEN.finditer(text))
    marked = set()
    for (a, b), flag in zip(token_spans, changed):
        if flag:
            marked.update(k for k, w in enumerate(words) if w.start() < b and w.end() > a)
    spans, run = [], []
    for k in sorted(marked):
        if run and k != run[-1] + 1:
            spans.append([words[run[0]].start(), words[run[-1]].end()])
            run = []
        run.append(k)
    if run:
        spans.append([words[run[0]].start(), words[run[-1]].end()])
    return spans


def make_record(row, tok, refill, rng, args, pool):
    """One corrupted record, or (None, reason). `refill(ids, positions)` returns
    `ids` with `positions` filled; it is the only place the model is used."""
    from classifier.inputs import SEPARATOR, encode, evidence_ids

    gold = row["references"][0]
    enc = tok(gold, add_special_tokens=False, return_offsets_mapping=True)
    gold_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(gold_ids) < 3:
        return None, "too_short"
    pieces = sentence_tokens(gold, offsets)
    if pieces is None:
        return None, "sentence_split"

    share = rng.uniform(args.min_noise, args.max_noise)
    summary_pos = pick_positions(len(gold_ids), share, rng)

    prefix = [] if tok.bos_token_id is None else [tok.bos_token_id]
    ev_ids, _ = evidence_ids(tok, [{"title": "Dialogue", "text": row["dialogue"]}])
    ev_start = len(prefix)
    prefix = prefix + ev_ids + encode(tok, SEPARATOR)
    for p in rng.sample(range(len(ev_ids)), round(share * len(ev_ids))):
        prefix[ev_start + p] = args.mask_id

    ids = prefix + gold_ids
    filled = refill(ids, [len(prefix) + p for p in summary_pos])
    corrupt = filled[len(prefix):]
    changed = same_text_unchanged(tok, corrupt, gold_ids)

    source, labels = [], []
    for j, (_, a, b) in enumerate(pieces):
        decoded = decode_spans(tok, corrupt[a:b])
        if decoded is None:
            return None, "decode"
        text, spans = decoded
        lead = len(text) - len(text.lstrip())
        text = text.strip()
        if not text:
            return None, "empty_sentence"
        spans = [(max(s - lead, 0), min(e - lead, len(text))) for s, e in spans]
        stale = stale_word_spans(text, spans, changed[a:b])
        source.append(text)
        labels.append({"label": "EDIT", "target_idx": j, "stale_spans": stale} if stale
                      else {"label": "KEEP", "target_idx": j})

    if pool and rng.random() < args.drop_prob:
        other = rng.choice(pool)
        if other[0] != row["id"]:
            k = rng.randrange(len(source) + 1)
            source.insert(k, other[1])
            labels.insert(k, {"label": "DROP", "target_idx": None, "similarity": 0.0})

    if not args.keep_unchanged and all(lab["label"] == "KEEP" for lab in labels):
        return None, "unchanged"
    return {
        "id": corrupt_id(row),
        "source_sentences": source,
        "target_sentences": [{"text": s} for s, _, _ in pieces],
        "labels": labels,
        "additions": [],
        "evidence": [{"title": "Dialogue", "text": row["dialogue"]}],
        "references": row["references"],
        "noise": round(share, 3),
    }, "kept"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="folder holding the official dialogsum .jsonl files")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None, help="default is classifier.inputs.MODEL")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--min-noise", type=float, default=0.15)
    parser.add_argument("--max-noise", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=8, help="sampler steps per summary")
    parser.add_argument("--temperature", type=float, default=0.7, help="0 takes the arg-max token")
    parser.add_argument("--drop-prob", type=float, default=0.0)
    parser.add_argument("--keep-unchanged", action="store_true")
    parser.add_argument("--limit", type=int, help="first N dialogues only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    from classifier.inputs import MODEL
    from infilling.model import MASK_ID, Model
    from infilling.sampler import sample

    args.mask_id = MASK_ID
    name = args.model or MODEL
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = Model(name, args.quant, tok)
    generator = torch.Generator(device=model.device)
    generator.manual_seed(args.seed)

    def refill(ids, positions):
        x = torch.tensor(ids, device=model.device)
        out, _ = sample(model, x, torch.tensor(positions, device=model.device), args.steps, args.temperature, generator)
        return out.tolist()

    rows, dropped = read_split(args.data, args.split)
    rows = rows[:args.limit]
    pool = [(r["id"], s) for r in rows for s in sentences(r["references"][0])] if args.drop_prob else []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    tally = {}
    with open(out, "a", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            if corrupt_id(row) in done:
                reason = "already_done"
            else:
                record, reason = make_record(row, tok, refill, random.Random(f"{args.seed}-{row['id']}"), args, pool)
                if record:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
            tally[reason] = tally.get(reason, 0) + 1
            if i % 100 == 0 or i == len(rows):
                print(f"{i}/{len(rows)}  {tally}", flush=True)
    print(f"dropped while reading: {dropped}")


if __name__ == "__main__":
    main()
