"""Run one infilling round over a collated split with one configuration.

Sentence labels are the derived ones (the oracle sentence labels of
Section 9.3). DROP sentences are deleted, and every EDIT sentence is
repaired left to right, so later sentences see earlier repairs. The masked
tokens come from --masks.

    oracle     exactly the derived stale tokens (oracle token masks). With
               --no-evidence this is the Risk 1 check of Section 11.
    predicted  the staleness head's scores on one frozen pass over E + A,
               through the masking policy of Section 7.1

Every ablation axis and tunable value is a flag, and one run is one
configuration. Defaults are the paper's defaults. The deletion margin
default is a placeholder until it is tuned on validation.

Output directory:
    config.json    every flag value. A rerun with a different one refuses.
    results.jsonl  one line per article: source and output sentences, every
                   candidate with its scores, and counters. A rerun resumes
                   after the last finished article.

Usage:
    python -m infilling.run ../datasets/collated/test.jsonl runs/oracle --masks oracle
    python -m infilling.run ../datasets/collated/test.jsonl runs/oracle-noev --masks oracle --no-evidence
    python -m infilling.run ../datasets/collated/val.jsonl runs/val-lam03 --masks predicted \
        --staleness-head heads/staleness/head.pt --lam 0.3 --limit 200
"""

import argparse
import itertools
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from classifier.extract import select
from classifier.inputs import DROP, EDIT, LABELS, MODEL, build
from infilling.model import Model
from infilling.repair import Settings, repair_sentence
from infilling.score import VERIFIER, Verifier
from infilling.spans import Article, oracle_spans, policy_spans

C_MAX = 3


def parse():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--masks", choices=["oracle", "predicted"], default="oracle")
    parser.add_argument("--staleness-head", help="head.pt from staleness.train, for --masks predicted")
    parser.add_argument("--tau", type=float, help="mask threshold; default the head's tuned tau")
    parser.add_argument("--c-max", type=int, default=C_MAX, help="remask budget per token")
    parser.add_argument("--evidence", action=argparse.BooleanOptionalAction, default=True,
                        help="--no-evidence removes the evidence from the input and the verifier")
    parser.add_argument("--lengths", choices=["same", "keep-or-drop", "full"], default="full",
                        help="candidate lengths: {l}, {0, l}, or {0, l/2, l, l+2, 2l}")
    parser.add_argument("--multi-span", choices=["sequential", "shared-ratio"], default="sequential")
    parser.add_argument("--steps", type=int, default=8, help="sampler steps T")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--scorer", choices=["pll", "verifier", "combined"], default="combined")
    parser.add_argument("--lam", type=float, default=0.5, help="weight of PLL in the combined score")
    parser.add_argument("--deletion-margin", type=float, default=0.1, help="0 turns the margin off")
    parser.add_argument("--pll-context", choices=["full", "local"], default="full")
    parser.add_argument("--pll-window", type=int, default=0, help="probe tokens within w of a span; 0 = all")
    parser.add_argument("--pll-norm", choices=["raw", "exp"], default="raw")
    parser.add_argument("--verifier", default=VERIFIER)
    parser.add_argument("--support-index", type=int, help="SUPPORTS class index, if the label names are generic")
    parser.add_argument("--batch", type=int, default=8, help="PLL probes per forward pass")
    parser.add_argument("--sample", type=int, help="random subset of articles")
    parser.add_argument("--limit", type=int, help="first N articles, after --sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.masks == "predicted" and not args.staleness_head:
        parser.error("--masks predicted needs --staleness-head")
    if args.scorer == "verifier" and not args.evidence:
        parser.error("--scorer verifier has nothing to verify against with --no-evidence")
    return args


def settings_of(args):
    return Settings(lengths=args.lengths, multi_span=args.multi_span, steps=args.steps,
                    temperature=args.temperature, scorer=args.scorer, lam=args.lam,
                    deletion_margin=args.deletion_margin, pll_context=args.pll_context,
                    pll_window=args.pll_window, pll_norm=args.pll_norm, batch=args.batch)


def load_head(path, device):
    ckpt = torch.load(path, map_location=device)
    weight = ckpt["state_dict"]["weight"]
    head = torch.nn.Linear(weight.shape[1], 1).to(device)
    head.load_state_dict(ckpt["state_dict"])
    return head.eval(), ckpt["mean"].to(device), ckpt["std"].to(device), ckpt["layer"], ckpt["tau"]


def stale_scores(model, example, head, mean, std, layer):
    """Token staleness pi_i (Eq. 7) for every sentence, from one frozen pass."""
    ids = torch.from_numpy(example.input_ids.astype(np.int64))[None].to(model.device)
    model.passes["score"] += 1
    hidden = model.hidden(ids, layer)[0].float()
    with torch.inference_mode():
        pi = head((hidden - mean) / std).sigmoid()[:, 0].cpu().numpy()
    return [pi[a:b] for a, b in example.spans]


def done_ids(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def main():
    args = parse()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "data": str(Path(args.data).resolve())}
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise SystemExit(f"{out} holds a run with a different configuration")
    else:
        config_path.write_text(json.dumps(config, indent=2))
    results_path = out / "results.jsonl"
    finished = done_ids(results_path)

    settings = settings_of(args)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = Model(args.model, args.quant, tok)
    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    verifier = None
    if args.evidence and args.scorer != "pll":
        verifier = Verifier(args.verifier, model.device, args.support_index)
    head = None
    if args.masks == "predicted":
        head, mean, std, layer, tuned_tau = load_head(args.staleness_head, model.device)
        tau = tuned_tau if args.tau is None else args.tau
        print(f"staleness head at layer {layer}, tau {tau}")

    records = list(itertools.islice(select(args.data, args.sample, args.seed), args.limit))
    todo = [r for r in records if r["id"] not in finished]
    print(f"{len(records)} articles, {len(finished)} already done, "
          f"masks {args.masks}, evidence {args.evidence}, scorer {args.scorer}")

    totals, t0 = Counter(), time.time()
    with open(results_path, "a") as results:
        for n, record in enumerate(todo, 1):
            start, stats = time.time(), Counter()
            model.passes.clear()
            source = record if args.evidence else {**record, "evidence": []}
            example = build(source, tok)
            evidence = [f"{s['title']}: {s['text']}" for s in source["evidence"]]
            article = Article.from_example(example, tok.bos_token_id)

            if head is not None:
                scores = stale_scores(model, example, head, mean, std, layer)
            spans = {}
            for j, label in enumerate(example.labels):
                if label == EDIT:
                    counts = article.counts[j]
                    spans[j] = (policy_spans(scores[j], counts, tau, args.c_max, stats) if head is not None
                                else oracle_spans(example.stale[j], counts, args.c_max))
            for j, label in enumerate(example.labels):
                if label == DROP:
                    article.drop(j)

            edits = []
            for j, sentence_spans in spans.items():
                if not sentence_spans:
                    stats["no_mask"] += 1
                    continue
                stats["sentences"] += 1
                masked = [tok.decode(article.sentences[j][a:b].tolist()) for a, b in sentence_spans]
                decisions = repair_sentence(model, verifier, article, j, sentence_spans, evidence,
                                            tok, settings, generator, stats)
                edits.append({"sentence": j, "spans": [list(s) for s in sentence_spans],
                              "masked": masked, "decisions": decisions})

            stats.update({f"passes_{k}": v for k, v in model.passes.items() if k != "check"})
            if verifier is not None:
                stats["verifier_pairs"] = verifier.pairs
                verifier.pairs = 0
            line = {
                "id": record["id"],
                "labels": [LABELS[y] for y in example.labels],
                "source": [s.strip() for s in record["source_sentences"]],
                "output": [tok.decode(s.tolist()).strip() for s in article.sentences],
                "edits": edits,
                "stats": dict(stats),
                "seconds": round(time.time() - start, 2),
            }
            results.write(json.dumps(line) + "\n")
            results.flush()
            totals.update(stats)

            rate = n / (time.time() - t0)
            print(f"{len(finished) + n}/{len(records)}  {1 / rate:.1f} s/article  "
                  f"eta {(len(todo) - n) / rate / 3600:.1f} h  spans {totals['spans']}  "
                  f"deletion wins {totals['deletion_wins']}/{totals['deletion_candidates']}", flush=True)


if __name__ == "__main__":
    main()
