"""Fine-tuned seq2seq editor baseline in the style of EDIT5 (Iv et al. 2022, FRUIT).

The model reads the outdated article and the evidence and writes the updated
article. An unchanged sentence is written as a copy token, so the model only
spells out the sentences it changes.

    input   update: [0] <sentence 0> [1] <sentence 1> ... evidence: (0) <title>: <text> (1) ...
    target  for each source sentence j in order, "[j]" for KEEP, "#j <target>"
            for EDIT and nothing for DROP

The input is cut at --max-input tokens from the end, so the evidence is cut
first and the article only when it is longer than the budget on its own. The
T5 sentencepiece vocabulary maps "<" and "{" to <unk>, so an edit is marked
"#j" and not "<j>". "[j]" and "#j" survive tokenization and decoding.

A generated text maps back to one output sentence per source sentence, the
format of infilling.run: "[j]" copies source sentence j, "#j text" replaces
it, and a missing j is a deletion, kept as an empty string so the output stays
aligned to the source and infilling.metrics --cascades scores it. Markers are
read only in increasing order of j. A malformed piece copies the source
sentence and counts as a fallback, with its reason:

    preamble        text before the first marker, ignored
    copy_text       text after a "[j]", ignored
    empty_edit      "#j" with no text
    truncated       generation hit --max-new-tokens, so the last edit and
                    every later sentence are copied
    no_marker       non-empty text with no marker, every sentence copied

The vocabulary also lacks some letters, such as ø, ð and í, and about 4% of
the train targets lose a character in a round trip through the tokenizer. An
edited sentence gets those letters back from the input: a word of the output
equal to a word of the article or evidence with its missing letters removed
is replaced by that word. The count of replaced words is in metrics.json.

Training fits on --train, computes the val loss after each epoch, keeps the
checkpoint with the lowest val loss in <out>/best and stops after --patience
epochs without improvement. Prediction decodes every --test article greedily
with the best checkpoint. A rerun skips training once it has finished and
resumes prediction after the last written article.

Outputs in --out:
    best/          checkpoint and tokenizer with the lowest val loss
    config.json    every flag, the model and the article counts
    results.jsonl  one {"id", "output", "raw", "fallbacks"} line per test article,
                   raw the generated text
    metrics.json   train time, val loss per epoch, best val loss, epochs run,
                   fallbacks, restored words, mean output length, seconds per
                   article

Usage:
    python -m experiments.editor --train ../datasets/synthetic/split/train.jsonl \\
        --val ../datasets/synthetic/split/val.jsonl --test ../datasets/synthetic/split/test.jsonl \\
        --out runs/editor
    python -m experiments.editor --test ../datasets/synthetic/split/test.jsonl --out runs/editor --predict-only
"""

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from classifier.inputs import read_jsonl

MODEL = "google/flan-t5-large"
PREDICT_FLAGS = ("test", "max_input", "max_new_tokens", "predict_batch", "predict_only", "limit")
MARKER = re.compile(r"(?<!\S)(?:\[(\d+)\]|#(\d+))(?!\S)")


def input_text(record):
    article = " ".join(f"[{j}] {s.strip()}" for j, s in enumerate(record["source_sentences"]))
    evidence = " ".join(f"({i}) {s['title']}: {s['text']}" for i, s in enumerate(record["evidence"]))
    return f"update: {article}", f" evidence: {evidence}"


def target_text(record):
    parts = []
    for j, entry in enumerate(record["labels"]):
        if entry["label"] == "KEEP":
            parts.append(f"[{j}]")
        elif entry["label"] == "EDIT":
            parts.append(f"#{j} {entry['target'].strip()}")
    return " ".join(parts)


def encode_input(tok, record, max_input):
    """Input ids ending in EOS, cut to max_input, the untruncated length and the part cut, if any."""
    article, evidence = (tok(t, add_special_tokens=False)["input_ids"] for t in input_text(record))
    ids = article + evidence
    room = max_input - 1
    cut = None if len(ids) <= room else "article" if len(article) > room else "evidence"
    return ids[:room] + [tok.eos_token_id], len(ids) + 1, cut


class Restorer:
    """Puts back letters the tokenizer cannot produce, from the words of the input."""

    def __init__(self, tok):
        self.tok = tok
        self.lost = {}

    def missing(self, ch):
        if ch not in self.lost:
            self.lost[ch] = self.tok.unk_token_id in self.tok(ch, add_special_tokens=False)["input_ids"]
        return self.lost[ch]

    def table(self, record):
        """Word with its missing letters removed -> word, for the article and evidence words that lose letters."""
        texts = record["source_sentences"] + [f"{s['title']} {s['text']}" for s in record["evidence"]]
        words = sorted({w for t in texts for w in re.findall(r"\w+", t)})
        present, table = set(words), {}
        for word in words:
            key = "".join(c for c in word if not self.missing(c))
            # A key that is itself an input word is left alone.
            if key and key != word and key not in present:
                table.setdefault(key, word)
        return table


def restore(text, table):
    count = 0

    def swap(m):
        nonlocal count
        if m.group() in table:
            count += 1
            return table[m.group()]
        return m.group()

    return re.sub(r"\w+", swap, text), count


def parse(raw, source, table=None, truncated=False):
    """Output sentences aligned to the source, the fallbacks taken and the count of restored words."""
    source = [s.strip() for s in source]
    n = len(source)
    marks, last = [], -1
    for m in MARKER.finditer(raw):
        j = int(m.group(1) if m.group(1) is not None else m.group(2))
        if last < j < n:
            marks.append((j, m.group(1) is None, m.start(), m.end()))
            last = j
    if not marks:
        if raw.strip():
            return list(source), [{"sentence": None, "reason": "no_marker"}], 0
        return [""] * n, [], 0

    output, fallbacks, restored = [""] * n, [], 0
    if raw[:marks[0][2]].strip():
        fallbacks.append({"sentence": None, "reason": "preamble"})
    for k, (j, edit, _, end) in enumerate(marks):
        text = raw[end:marks[k + 1][2] if k + 1 < len(marks) else len(raw)].strip()
        if not edit:
            output[j] = source[j]
            if text:
                fallbacks.append({"sentence": j, "reason": "copy_text"})
        elif truncated and k == len(marks) - 1:
            output[j] = source[j]
            fallbacks.append({"sentence": j, "reason": "truncated"})
        elif text:
            output[j], count = restore(text, table or {})
            restored += count
        else:
            output[j] = source[j]
            fallbacks.append({"sentence": j, "reason": "empty_edit"})
    if truncated:
        for j in range(last + 1, n):
            output[j] = source[j]
            fallbacks.append({"sentence": j, "reason": "truncated"})
    return output, fallbacks, restored


def percentiles(name, lengths, budget):
    lengths = np.asarray(lengths)
    cells = "  ".join(f"p{p} {np.percentile(lengths, p):.0f}" for p in (50, 90, 95, 99))
    print(f"{name} tokens: mean {lengths.mean():.0f}  {cells}  max {lengths.max()}  "
          f"over {budget}: {(lengths > budget).sum()} of {len(lengths)}", flush=True)
    return {f"p{p}": float(np.percentile(lengths, p)) for p in (50, 90, 95, 99)} | {
        "mean": float(lengths.mean()), "max": int(lengths.max()), "over": int((lengths > budget).sum())}


def encode_split(tok, records, args, report=False):
    """(input ids, label ids) tensors per article. With report, prints and returns length percentiles."""
    examples, in_lengths, out_lengths, cuts = [], [], [], Counter()
    for record in records:
        ids, length, cut = encode_input(tok, record, args.max_input)
        target = tok(target_text(record))["input_ids"]
        in_lengths.append(length)
        out_lengths.append(len(target))
        cuts[cut] += 1
        if len(target) > args.max_new_tokens:
            target = target[:args.max_new_tokens - 1] + [tok.eos_token_id]
        examples.append((torch.tensor(ids), torch.tensor(target)))
    if not report:
        return examples, None
    stats = {"input": percentiles("input", in_lengths, args.max_input),
             "target": percentiles("target", out_lengths, args.max_new_tokens),
             "evidence_cut": cuts["evidence"], "article_cut": cuts["article"]}
    print(f"cut inputs: {cuts['evidence']} lose evidence, {cuts['article']} lose article text", flush=True)
    return examples, stats


def batches(examples, size, rng):
    """Batches of similar input length in random order, so padding stays small."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    chunk = size * 50
    groups = []
    for i in range(0, len(order), chunk):
        part = sorted(order[i:i + chunk], key=lambda k: len(examples[k][0]))
        groups += [part[k:k + size] for k in range(0, len(part), size)]
    rng.shuffle(groups)
    return groups


def collate(examples, idx, pad, device):
    inputs = pad_sequence([examples[k][0] for k in idx], batch_first=True, padding_value=pad).to(device)
    labels = pad_sequence([examples[k][1] for k in idx], batch_first=True, padding_value=-100).to(device)
    return inputs, inputs != pad, labels


def loss_sum(model, inputs, mask, labels, cuda):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
        logits = model(input_ids=inputs, attention_mask=mask, labels=labels).logits
    return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")


@torch.no_grad()
def val_loss(model, examples, args, pad, device):
    model.eval()
    order = sorted(range(len(examples)), key=lambda k: len(examples[k][0]))
    total, tokens = 0.0, 0
    for i in range(0, len(order), args.predict_batch):
        inputs, mask, labels = collate(examples, order[i:i + args.predict_batch], pad, device)
        total += loss_sum(model, inputs, mask, labels, device.type == "cuda").item()
        tokens += int((labels != -100).sum())
    model.train()
    return total / tokens


def train(args, tok, device, out, metrics):
    train_records = list(read_jsonl(args.train))[:args.limit]
    val_records = list(read_jsonl(args.val))[:args.limit]
    print(f"train {len(train_records)} articles, val {len(val_records)}", flush=True)
    train_set, metrics["train_lengths"] = encode_split(tok, train_records, args, report=True)
    val_set, _ = encode_split(tok, val_records, args)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    decay = [p for name, p in model.named_parameters() if "layer_norm" not in name]
    rest = [p for name, p in model.named_parameters() if "layer_norm" in name]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}],
                                  lr=args.lr)
    per_epoch = math.ceil(math.ceil(len(train_set) / args.batch) / args.accum)
    total_steps = per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    rng = random.Random(args.seed)
    cuda = device.type == "cuda"
    print(f"{per_epoch} optimizer steps per epoch, {total_steps} at most, "
          f"{'bf16 autocast' if cuda else 'fp32'} on {device}", flush=True)

    best, stale, history, step, t0 = math.inf, 0, [], 0, time.time()
    for epoch in range(1, args.epochs + 1):
        groups = batches(train_set, args.batch, rng)
        running, seen = 0.0, 0
        for g in range(0, len(groups), args.accum):
            group = groups[g:g + args.accum]
            tokens = sum(int(len(train_set[k][1])) for idx in group for k in idx)
            for idx in group:
                inputs, mask, labels = collate(train_set, idx, tok.pad_token_id, device)
                loss = loss_sum(model, inputs, mask, labels, cuda) / tokens
                loss.backward()
                running += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            seen += 1
            if step % args.log_every == 0 or g + args.accum >= len(groups):
                rate = step / (time.time() - t0)
                print(f"epoch {epoch}  step {step}/{total_steps}  loss {running / seen:.4f}  "
                      f"{rate:.2f} steps/s  eta {(total_steps - step) / rate / 3600:.2f} h "
                      f"(epoch {(epoch * per_epoch - step) / rate / 60:.1f} min)", flush=True)
        loss = val_loss(model, val_set, args, tok.pad_token_id, device)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": loss})
        print(f"epoch {epoch}  val loss {loss:.4f}  best {min(best, loss):.4f}", flush=True)
        if loss < best:
            best, stale = loss, 0
            model.save_pretrained(out / "best")
            tok.save_pretrained(out / "best")
        else:
            stale += 1
        metrics.update({"val_history": history, "best_val_loss": best, "epochs_run": epoch,
                        "best_epoch": min(history, key=lambda h: h["val_loss"])["epoch"]})
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        if stale >= args.patience:
            print(f"no improvement for {stale} epoch(s), stopping", flush=True)
            break
    metrics["train_seconds"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    del model, optimizer
    if cuda:
        torch.cuda.empty_cache()
    return len(train_records), len(val_records)


def done_ids(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


@torch.no_grad()
def predict(args, device, out, records):
    tok = AutoTokenizer.from_pretrained(out / "best")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(out / "best", dtype=dtype).to(device).eval()
    restorer = Restorer(tok)
    results_path = out / "results.jsonl"
    finished = done_ids(results_path)
    todo = [r for r in records if r["id"] not in finished]
    encoded = sorted(((r, encode_input(tok, r, args.max_input)[0]) for r in todo), key=lambda x: -len(x[1]))
    print(f"{len(records)} test articles, {len(finished)} already done", flush=True)

    t0 = time.time()
    with open(results_path, "a") as results:
        for i in range(0, len(encoded), args.predict_batch):
            chunk = encoded[i:i + args.predict_batch]
            inputs = pad_sequence([torch.tensor(ids) for _, ids in chunk], batch_first=True,
                                  padding_value=tok.pad_token_id).to(device)
            generated = model.generate(input_ids=inputs, attention_mask=inputs != tok.pad_token_id,
                                       max_new_tokens=args.max_new_tokens, num_beams=1, do_sample=False)
            for (record, _), seq in zip(chunk, generated.tolist()):
                seq = seq[1:]
                truncated = tok.eos_token_id not in seq
                raw = tok.decode(seq, skip_special_tokens=True).strip()
                output, fallbacks, restored = parse(raw, record["source_sentences"], restorer.table(record), truncated)
                line = {"id": record["id"], "output": output, "raw": raw, "fallbacks": fallbacks, "restored": restored}
                results.write(json.dumps(line) + "\n")
            results.flush()
            n = i + len(chunk)
            rate = n / (time.time() - t0)
            print(f"{len(finished) + n}/{len(records)}  {1 / rate:.2f} s/article  "
                  f"eta {(len(encoded) - n) / rate / 60:.1f} min", flush=True)
    return len(encoded), time.time() - t0


def summarize(out, records, metrics, predicted, seconds):
    """Prediction metrics over every line of results.jsonl."""
    sources = {r["id"]: r["source_sentences"] for r in records}
    lines = [line for line in read_jsonl(out / "results.jsonl") if line["id"] in sources]
    reasons, decisions = Counter(), Counter()
    for line in lines:
        reasons.update(f["reason"] for f in line["fallbacks"])
        for s, o in zip(sources[line["id"]], line["output"]):
            decisions["deleted" if not o else "copied" if o == s.strip() else "rewritten"] += 1
    metrics.update({
        "articles": len(lines),
        "fallbacks": sum(reasons.values()),
        "fallback_reasons": dict(reasons),
        "articles_with_fallback": sum(bool(line["fallbacks"]) for line in lines),
        "restored_words": sum(line["restored"] for line in lines),
        "sentences": dict(decisions),
        "mean_output_words": float(np.mean([len(" ".join(line["output"]).split()) for line in lines])),
        "mean_output_sentences": float(np.mean([sum(bool(o) for o in line["output"]) for line in lines])),
        "mean_raw_chars": float(np.mean([len(line["raw"]) for line in lines])),
    })
    if predicted:
        metrics["seconds_per_article"] = seconds / predicted
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("val_history", "train_lengths")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", help="dataset JSONL to fit on")
    parser.add_argument("--val", help="dataset JSONL for the val loss and early stopping")
    parser.add_argument("--test", required=True, help="dataset JSONL decoded once with the best checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--predict-only", action="store_true", help="load <out>/best and only predict")
    parser.add_argument("--max-input", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=384, help="also the cap on train target length")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--predict-batch", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="trade speed for memory")
    parser.add_argument("--log-every", type=int, default=50, help="optimizer steps between log lines")
    parser.add_argument("--limit", type=int, help="first N articles of every split, for a smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.predict_only and not (args.train and args.val):
        parser.error("training needs --train and --val")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path, config_path = out / "metrics.json", out / "config.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    trained = "train_seconds" in metrics and (out / "best").exists()
    if args.predict_only and not (out / "best").exists():
        raise SystemExit(f"{out / 'best'} does not exist")
    if not args.predict_only and not trained:
        if (out / "results.jsonl").exists():
            raise SystemExit(f"{out} holds predictions of an earlier model; use a new --out")
        metrics = {}
        tok = AutoTokenizer.from_pretrained(args.model)
        n_train, n_val = train(args, tok, device, out, metrics)
        config = {**vars(args), "n_train": n_train, "n_val": n_val}
    elif trained:
        print(f"training finished earlier, best val loss {metrics['best_val_loss']:.4f}", flush=True)

    records = list(read_jsonl(args.test))[:args.limit]
    # Training values stay, prediction values are the ones of this call.
    config.update({k: v for k, v in vars(args).items() if k not in config or k in PREDICT_FLAGS})
    config["n_test"] = len(records)
    config_path.write_text(json.dumps(config, indent=2))
    predicted, seconds = predict(args, device, out, records)
    summarize(out, records, metrics, predicted, seconds)


if __name__ == "__main__":
    main()
