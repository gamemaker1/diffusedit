"""Cache mean-pooled LLaDA hidden states for every source sentence (Eq. 4).

One frozen forward pass per article over E ⊕ A, no masks. Forward hooks read
the chosen blocks. The deepest hook stops the pass, so later blocks and the
LM head never run. The defaults fit an 11 GB RTX 2080 Ti: 4-bit NF4 weights,
fp16 compute (Turing has no native bf16), batch size 1.

Output directory:
    config.json    model, quantization, layers, source file
    ids.json       article ids in row order
    labels.npy     int8 [n_sentences], KEEP/EDIT/DROP
    article.npy    int32 [n_sentences], index into ids.json
    layer{L}.npy   float16 [n_sentences, d], pooled output of block L (1-based)
    progress.json  articles and rows written so far. A rerun resumes here.

Usage:
    python -m classifier.extract ../datasets/collated/test.jsonl features/test
    python -m classifier.extract ../datasets/collated/train.jsonl features/train --sample 20000
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    examples = [build(r, tok) for r in select(args.data, args.sample, args.seed)]
    offsets = np.cumsum([0] + [len(e.spans) for e in examples])
    rows = int(offsets[-1])

    model = load_model(args.model, args.quant)
    blocks_name, blocks = find_blocks(model)
    n_layers = len(blocks)
    layers = sorted(set(args.layers or [round(0.75 * n_layers), n_layers]))
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
    print(f"{len(examples)} articles, {rows} sentences, layers {layers} of {n_layers}, resuming at {start}")

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
