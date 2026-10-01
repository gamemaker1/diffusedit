"""Cache mean-pooled LLaDA hidden states for every source sentence (Eq. 4).

One frozen forward pass per article over E ⊕ A, no masks. Forward hooks read
the chosen blocks. The deepest hook stops the pass, so later blocks and the
LM head never run. The defaults fit an 11 GB RTX 2080 Ti: 4-bit NF4 weights,
fp16 compute (Turing has no native bf16), batch size 1.

The same pass also stores the unpooled hidden state of every token inside an
EDIT sentence, from one layer, with its stale label, for the token staleness
head (Eq. 7).

Output directory:
    config.json    model, quantization, layers, source file
    ids.json       article ids in row order
    labels.npy     int8 [n_sentences], KEEP/EDIT/DROP
    article.npy    int32 [n_sentences], index into ids.json
    layer{L}.npy   float16 [n_sentences, d], pooled output of block L (1-based)
    progress.json  articles and rows written so far. A rerun resumes here.
    tokens/layer{L}.npy  float16 [n_tokens, d], one row per EDIT-sentence token
    tokens/labels.npy    int8 [n_tokens], 1 if the token is stale
    tokens/row.npy       int32 [n_tokens], the token's sentence row

Usage:
    python -m classifier.extract ../datasets/collated/train.jsonl features/train --sample 20000 --seed 42
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from classifier.inputs import MODEL, build, read_jsonl

FLUSH_EVERY = 200


class Stop(Exception):
    """Raised by the deepest hook to end the forward pass early."""


def load_model(name, quant):
    kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16, "device_map": {"": 0}}
    if quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    return AutoModel.from_pretrained(name, **kwargs).eval()


def find_blocks(model):
    """The ModuleList of transformer blocks, found by its length."""
    config = model.config
    n = getattr(config, "n_layers", None) or getattr(config, "num_hidden_layers")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == n:
            return name, module
    raise RuntimeError(f"no ModuleList of length {n} in {type(model).__name__}")


def attach(blocks, layers, store):
    deepest = max(layers)
    handles = []
    for layer in layers:
        def hook(module, args, output, layer=layer):
            store[layer] = output[0] if isinstance(output, tuple) else output
            if layer == deepest:
                raise Stop
        handles.append(blocks[layer - 1].register_forward_hook(hook))
    return handles


def select(path, sample, seed):
    """Stream records, keeping a seeded random subset of `sample` articles."""
    if not sample:
        return read_jsonl(path)
    with open(path) as f:
        total = sum(1 for line in f if line.strip())
    keep = set(random.Random(seed).sample(range(total), min(sample, total)))
    return (r for i, r in enumerate(read_jsonl(path)) if i in keep)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--layers", type=int, nargs="*", help="1-based blocks; default 3/4 depth and final")
    parser.add_argument("--sample", type=int, help="random subset of articles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-layer", type=int, help="1-based block for token states; default final")
    parser.add_argument("--tokens", action=argparse.BooleanOptionalAction, default=True,
                        help="store EDIT-sentence token states")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    examples = [build(r, tok) for r in select(args.data, args.sample, args.seed)]
    offsets = np.cumsum([0] + [len(e.spans) for e in examples])
    rows = int(offsets[-1])
    evidence_offsets = np.cumsum([0] + [len(e.evidence_spans) for e in examples])
    evidence_rows = int(evidence_offsets[-1])
    token_counts = [sum(len(m) for m in e.stale if m is not None) for e in examples]
    token_offsets = np.cumsum([0] + token_counts) if args.tokens else np.zeros(len(examples) + 1, dtype=np.int64)
    n_tokens = int(token_offsets[-1])

    model = load_model(args.model, args.quant)
    blocks_name, blocks = find_blocks(model)
    n_layers = len(blocks)
    token_layer = (args.token_layer or n_layers) if args.tokens else None
    layers = sorted(set(args.layers or [round(0.75 * n_layers), n_layers]) | ({token_layer} - {None}))
    d = model.config.d_model if hasattr(model.config, "d_model") else model.config.hidden_size

    config = {
        "model": args.model,
        "quant": args.quant,
        "layers": layers,
        "n_layers": n_layers,
        "blocks": blocks_name,
        "d": d,
        "data": str(Path(args.data).resolve()),
        "sample": args.sample,
        "seed": args.seed,
        "articles": len(examples),
        "rows": rows,
        "evidence_rows": evidence_rows,
        "token_layer": token_layer,
        "tokens": n_tokens,
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
        np.save(out / "labels.npy", np.concatenate([e.labels for e in examples]).astype(np.int8))
        np.save(out / "article.npy", np.repeat(np.arange(len(examples), dtype=np.int32), np.diff(offsets)))
        np.save(out / "evidence_article.npy", np.repeat(np.arange(len(examples), dtype=np.int32), np.diff(evidence_offsets)))
        if token_layer:
            (out / "tokens").mkdir(exist_ok=True)
            marks = [m for e in examples for m in e.stale if m is not None]
            np.save(out / "tokens" / "labels.npy", np.concatenate(marks).astype(np.int8))
            sentence_rows = [offsets[k] + j for k, e in enumerate(examples)
                             for j, m in enumerate(e.stale) if m is not None]
            np.save(out / "tokens" / "row.npy",
                    np.repeat(np.asarray(sentence_rows, dtype=np.int32), [len(m) for m in marks]))
        progress = {"articles": 0, "rows": 0, "evidence_rows": 0, "tokens": 0, "nonfinite": 0}
        mode = "w+"
    memmaps = {
        layer: np.lib.format.open_memmap(out / f"layer{layer}.npy", mode=mode, dtype=np.float16, shape=(rows, d))
        for layer in layers
    }
    evidence_memmaps = {
        layer: np.lib.format.open_memmap(out / f"evidence_layer{layer}.npy", mode=mode, dtype=np.float16, shape=(evidence_rows, d))
        for layer in layers
    }
    token_mm = None
    if token_layer:
        token_mm = np.lib.format.open_memmap(out / "tokens" / f"layer{token_layer}.npy", mode=mode,
                                             dtype=np.float16, shape=(n_tokens, d))

    def save_progress(k):
        for mm in [*memmaps.values(), *evidence_memmaps.values(), token_mm]:
            if mm is not None:
                mm.flush()
        progress["articles"] = k
        progress["rows"] = int(offsets[k])
        progress["evidence_rows"] = int(evidence_offsets[k])
        progress["tokens"] = int(token_offsets[k])
        progress_path.write_text(json.dumps(progress))

    store = {}
    attach(blocks, layers, store)
    device = next(model.parameters()).device
    start, t0 = progress["articles"], time.time()
    print(f"{len(examples)} articles, {rows} sentences, {n_tokens} EDIT tokens from layer {token_layer}, "
          f"layers {layers} of {n_layers}, seed {args.seed}, resuming at {start}")

    with torch.inference_mode():
        for k in range(start, len(examples)):
            example = examples[k]
            input_ids = torch.from_numpy(example.input_ids).long()[None].to(device)
            try:
                model(input_ids=input_ids)
            except Stop:
                pass
            for layer, mm in memmaps.items():
                hidden = store[layer][0].float()
                
                pooled = torch.stack([hidden[a:b].mean(0) for a, b in example.spans]).half()
                if not torch.isfinite(pooled).all():
                    progress["nonfinite"] += 1
                mm[offsets[k]:offsets[k + 1]] = pooled.cpu().numpy()
                
                if example.evidence_spans:
                    pooled_ev = torch.stack([hidden[a:b].mean(0) for a, b in example.evidence_spans]).half()
                    if not torch.isfinite(pooled_ev).all():
                        progress["nonfinite"] += 1
                    evidence_memmaps[layer][evidence_offsets[k]:evidence_offsets[k + 1]] = pooled_ev.cpu().numpy()
                    
            if token_mm is not None:
                hidden = store[token_layer][0]
                parts = [hidden[a:b] for (a, b), m in zip(example.spans, example.stale) if m is not None]
                if parts:
                    token_mm[token_offsets[k]:token_offsets[k + 1]] = torch.cat(parts).half().cpu().numpy()
            store.clear()

            done = k + 1
            if done % FLUSH_EVERY == 0 or done == len(examples):
                save_progress(done)
                rate = (done - start) / (time.time() - t0)
                eta = (len(examples) - done) / rate / 3600
                print(f"{done}/{len(examples)}  {rate:.2f} articles/s  eta {eta:.1f} h  "
                      f"nonfinite {progress['nonfinite']}", flush=True)


if __name__ == "__main__":
    main()
