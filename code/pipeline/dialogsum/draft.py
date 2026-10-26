"""Writes the stale draft summaries with a chat model (default Llama-3-8B-Instruct).

Reads the prompt file from `convert.py prompts` and writes one line per
dialogue, {"id": ..., "draft": ...}, which `convert.py build --drafts` reads.
Decoding is greedy, so a rerun gives the same drafts. Reruns skip ids already
in --out.

meta-llama/Meta-Llama-3-8B-Instruct is gated: accept the licence on its
Hugging Face page and log in with `huggingface-cli login` first.

Usage (from the pipeline folder, on a GPU):
    python -m dialogsum.draft work/test.prompts.jsonl work/test.drafts.jsonl
"""

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompts")
    parser.add_argument("out")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quant", choices=["4bit", "none"], default="none")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=96, help="new tokens per summary")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from classifier.extract import compute_dtype

    dtype = compute_dtype()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    kwargs = {"torch_dtype": dtype, "device_map": {"": 0}}
    if args.quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    stops = [tok.eos_token_id]
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0 and eot != tok.unk_token_id:
        stops.append(eot)

    with open(args.prompts, encoding="utf-8") as f:
        todo = [json.loads(line) for line in f if line.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [row for row in todo if row["id"] not in done]
    print(f"{len(todo)} drafts to write, {len(done)} already done", flush=True)

    with open(out, "a", encoding="utf-8") as f, torch.inference_mode():
        for i in range(0, len(todo), args.batch):
            rows = todo[i:i + args.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                             add_generation_prompt=True, tokenize=False) for r in rows]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 eos_token_id=stops, pad_token_id=tok.pad_token_id)
            for row, ids in zip(rows, gen[:, enc["input_ids"].shape[1]:]):
                draft = tok.decode(ids, skip_special_tokens=True).strip()
                f.write(json.dumps({"id": row["id"], "draft": draft}) + "\n")
            f.flush()
            print(f"{min(i + args.batch, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
