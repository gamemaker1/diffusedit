"""Cache mean-pooled LLaDA hidden states for every EVIDENCE snippet (not
sentences -- see classifier.extract for that), for the evidence-augmented
classifier experiment (experiments/evidence_train.py).

Self-contained: imports classifier.inputs.build / classifier.extract's model
utilities but modifies neither file. This means a SECOND forward pass over
the same articles classifier.extract already processed -- real, duplicated
GPU time, accepted here specifically to avoid touching classifier/extract.py
or classifier/inputs.py while Vedant is actively iterating on them. Once this
experiment is validated, folding evidence pooling into his single pass
(like the token scorer eventually was) is the efficient long-term fix.

Evidence spans are recomputed locally (evidence_spans_and_ids below),
replicating classifier.inputs.evidence_ids's exact tokenization + truncation
logic so the spans line up with where build() actually placed those same
tokens in example.input_ids. This is fragile by construction: if
classifier.inputs.evidence_ids changes, this must change with it -- there is
a round-trip check in main() that catches drift instead of silently
misaligning.

Output directory:
    config.json         model, quantization, layers, source file
    ids.json             article ids in row order
    article.npy           int32 [n_evidence_rows], index into ids.json
    layer{L}.npy          float16 [n_evidence_rows, d], pooled evidence-snippet vectors
    progress.json         articles and rows written so far. A rerun resumes here.

Usage:
    python -m experiments.evidence_extract ../datasets/collated/train.jsonl evidence_features/train
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from classifier.extract import Stop, attach, find_blocks, load_model, select
from classifier.inputs import EVIDENCE_BUDGET, MODEL, build, encode, read_jsonl


def evidence_spans(tok, evidence, budget=EVIDENCE_BUDGET):
    """Replicates classifier.inputs.evidence_ids's tokenization exactly, but
    keeps per-snippet spans instead of just the concatenated ids.
    """
    ids, spans = [], []
    for i, snippet in enumerate(evidence):
        text = f"{snippet['title']}: {snippet['text']}"
        piece = encode(tok, text if i == 0 else "\n" + text)
        start = len(ids)
        ids += piece
        spans.append((start, len(ids)))
    if len(ids) > budget:
        ids = ids[:budget]
        spans = [(s, min(e, budget)) for s, e in spans if s < budget]
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--layers", type=int, nargs="*", help="1-based blocks; default 3/4 depth and final")
    parser.add_argument("--sample", type=int, help="random subset of articles (use same value as classifier.extract)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    records = list(select(args.data, args.sample, args.seed))
    examples = [build(r, tok) for r in records]

    bos_offset = 0 if tok.bos_token_id is None else 1
    per_article_spans = []
    for record, example in zip(records, examples):
        spans = [(s + bos_offset, e + bos_offset) for s, e in evidence_spans(tok, record["evidence"])]
        # round-trip check: does the span this file computed actually decode
        # to the evidence text build() placed there? If evidence_ids ever
        # changes, this is what catches the drift instead of silently
        # pooling the wrong tokens.
        for (a, b), snippet in zip(spans, record["evidence"]):
            expected = f"{snippet['title']}: {snippet['text']}"
            decoded = tok.decode(example.input_ids[a:b].tolist())
            if "".join(decoded.split()) not in "".join(expected.split()):
                raise SystemExit(
                    f"{record['id']}: evidence span mismatch -- classifier.inputs.evidence_ids "
                    f"has likely changed since this file was written. expected substring of "
                    f"{expected!r}, decoded {decoded!r}"
                )
        per_article_spans.append(spans)

    offsets = np.cumsum([0] + [len(s) for s in per_article_spans])
    rows = int(offsets[-1])
    if rows == 0:
        raise SystemExit("no evidence snippets found across this split")

    model = load_model(args.model, args.quant)
    blocks_name, blocks = find_blocks(model)
    n_layers = len(blocks)
    layers = sorted(set(args.layers or [round(0.75 * n_layers), n_layers]))
    d = model.config.d_model if hasattr(model.config, "d_model") else model.config.hidden_size

    config = {
        "model": args.model, "quant": args.quant, "layers": layers, "n_layers": n_layers,
        "blocks": blocks_name, "d": d, "data": str(Path(args.data).resolve()),
        "sample": args.sample, "seed": args.seed,
        "articles": len(examples), "rows": rows,
    }
    progress_path = out / "progress.json"
    config_path = out / "config.json"
    if progress_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise SystemExit(f"{out} holds features from a different configuration")
        progress = json.loads(progress_path.read_text())
        mode = "r+"
    else:
        config_path.write_text(json.dumps(config, indent=2))
        (out / "ids.json").write_text(json.dumps([e.id for e in examples]))
        np.save(out / "article.npy", np.repeat(
            np.arange(len(examples), dtype=np.int32), [len(s) for s in per_article_spans]))
        progress = {"articles": 0, "rows": 0, "nonfinite": 0}
        mode = "w+"
    memmaps = {
        layer: np.lib.format.open_memmap(out / f"layer{layer}.npy", mode=mode, dtype=np.float16, shape=(rows, d))
        for layer in layers
    }

    def save_progress(k):
        for mm in memmaps.values():
            mm.flush()
        progress["articles"], progress["rows"] = k, int(offsets[k])
        progress_path.write_text(json.dumps(progress))

    store = {}
    attach(blocks, layers, store)
    device = next(model.parameters()).device
    start, t0 = progress["articles"], time.time()
    print(f"{len(examples)} articles, {rows} evidence snippets, layers {layers} of {n_layers}, "
          f"resuming at {start}")

    with torch.inference_mode():
        for k in range(start, len(examples)):
            example = examples[k]
            spans = per_article_spans[k]
            if spans:
                input_ids = torch.from_numpy(example.input_ids).long()[None].to(device)
                try:
                    model(input_ids=input_ids)
                except Stop:
                    pass
                for layer, mm in memmaps.items():
                    hidden = store[layer][0].float()
                    pooled = torch.stack([hidden[a:b].mean(0) for a, b in spans]).half()
                    if not torch.isfinite(pooled).all():
                        progress["nonfinite"] += 1
                    mm[offsets[k]:offsets[k + 1]] = pooled.cpu().numpy()
                store.clear()

            done = k + 1
            if done % 200 == 0 or done == len(examples):
                save_progress(done)
                rate = (done - start) / (time.time() - t0)
                eta = (len(examples) - done) / rate / 3600
                print(f"{done}/{len(examples)}  {rate:.2f} articles/s  eta {eta:.1f} h  "
                      f"nonfinite {progress['nonfinite']}", flush=True)


if __name__ == "__main__":
    main()
