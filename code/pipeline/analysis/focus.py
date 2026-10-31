"""Where the attention staleness head looks in the evidence.

The head of staleness.attention scores every token of an EDIT sentence after
attending over the evidence tokens and one null slot. This script reads the
attention weights of a saved head on a feature directory with evidence token
states and asks, token by token, what the weight falls on:

    null        the share of weight on the null slot
    new value   the share of weight on evidence tokens inside a new value of the
                sentence, the text that the gold rewrite puts in place of a
                stale span (experiments.stress.gold_spans), where the evidence
                holds that value
    top token   whether the evidence token with the most weight lies inside a
                new value of the sentence
    top passage whether the passage with the most weight holds a new value of
                the sentence, over articles with more than one passage

Every share is averaged over the heads, then over tokens, split into stale
and non-stale tokens. The uniform share of new-value tokens among the evidence
tokens is reported next to the new-value share as the chance level. The new
values are grouped by form: a four-digit year, another number, or a name.

The evidence token k of an article is the token at input position
evidence_spans[0][0] + k of classifier.inputs.build, so the passage and the
characters of every evidence token come from tokenizing the passages again as
classifier.inputs.evidence_ids does.

Usage, on the machine that holds the features:
    python -m analysis.focus FEATURES HEAD DATA --model MODEL --out focus.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from classifier.inputs import EVIDENCE_BUDGET, build, encode, read_jsonl
from experiments.stress import gold_spans
from staleness.attention import Articles, EvidenceAttention, collate


def form(value):
    value = value.strip()
    if re.fullmatch(r"\d{4}", value):
        return "year"
    if re.search(r"\d", value):
        return "number"
    return "name"


def evidence_tokens(tok, evidence, budget=EVIDENCE_BUDGET):
    """Passage index and (start, end) characters within its passage string of every evidence token."""
    passages, chars, texts = [], [], []
    for i, snippet in enumerate(evidence):
        text = f"{snippet['title']}: {snippet['text']}"
        text = text if i == 0 else "\n" + text
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        assert enc["input_ids"] == encode(tok, text)
        passages += [i] * len(enc["input_ids"])
        chars += enc["offset_mapping"]
        texts.append(text)
    return passages[:budget], chars[:budget], texts


def value_mask(values, passages, chars, texts):
    """True at every evidence token whose characters overlap an occurrence of a value."""
    mask = np.zeros(len(passages), dtype=bool)
    for value in values:
        pattern = re.compile(rf"(?<!\w){re.escape(value.strip().lower())}(?!\w)")
        spans = {i: [m.span() for m in pattern.finditer(t.lower())] for i, t in enumerate(texts)}
        for k, (i, (a, b)) in enumerate(zip(passages, chars)):
            if any(a < e and b > s for s, e in spans[i]):
                mask[k] = True
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("features", help="feature directory with tokens/evidence_layer{L}.npy")
    parser.add_argument("head", help="head.pt of staleness.attention")
    parser.add_argument("data", help="the split the features were extracted from")
    parser.add_argument("--model", required=True, help="tokenizer path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, help="first N articles, for a smoke test")
    parser.add_argument("--examples", type=int, default=12, help="stale tokens to print with their top evidence tokens")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    device = torch.device("cpu")
    ckpt = torch.load(args.head, map_location=device)
    head = EvidenceAttention(**ckpt["config"]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    norm = [ckpt[k].to(device) for k in ("mean", "std", "ev_mean", "ev_std")]
    data = Articles(args.features, ckpt["layer"])
    ids = json.loads((Path(args.features) / "ids.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    records = {r["id"]: r for r in read_jsonl(args.data)}
    row = np.load(Path(args.features) / "tokens" / "row.npy")
    article_of_row = np.load(Path(args.features) / "article.npy")

    rows = defaultdict(list)   # (stale, form) -> list of per-token dicts
    passage_hits = defaultdict(list)
    examples = []
    articles = sorted(data.tokens)[:args.limit]
    with torch.no_grad():
        for a in articles:
            record = records[ids[a]]
            example = build(record, tok)
            passages, chars, texts = evidence_tokens(tok, record["evidence"])
            if len(passages) != data.offsets[a + 1] - data.offsets[a]:
                raise SystemExit(f"{ids[a]}: {len(passages)} evidence tokens from the tokenizer, "
                                 f"{data.offsets[a + 1] - data.offsets[a]} stored")
            tokens, token_mask, evidence, evidence_mask, labels = collate(data, [a], norm, device)
            _, attn = head(tokens, evidence, evidence_mask, weights=True)
            attn = attn[0].mean(0)[token_mask[0]]          # [T, M + 1], heads averaged
            null, over = attn[:, 0].numpy(), attn[:, 1:].numpy()
            start, end = data.tokens[a]
            sentence_rows = row[start:end]
            # The sentence of each token: EDIT sentences of the article, in order.
            first_row = int(np.searchsorted(article_of_row, a))
            edit_sentences = [j for j, e in enumerate(record["labels"]) if e["label"] == "EDIT"]
            token_texts = []
            for j in edit_sentences:
                prefix = "" if j == 0 else " "
                piece = tok(prefix + record["source_sentences"][j].strip(), add_special_tokens=False)["input_ids"]
                token_texts += [(j, tok.decode([t])) for t in piece]
            if len(token_texts) != end - start:
                raise SystemExit(f"{ids[a]}: {len(token_texts)} sentence tokens from the tokenizer, {end - start} stored")
            masks, forms = {}, {}
            for j in edit_sentences:
                entry = record["labels"][j]
                pairs = gold_spans(record["source_sentences"][j], entry["target"], entry["stale_spans"])
                new = [n for _, n in pairs if n.strip()]
                masks[j] = value_mask(new, passages, chars, texts)
                forms[j] = form(new[0]) if new else "none"
            for t in range(end - start):
                j, text = token_texts[t]
                stale = bool(labels[t])
                m = masks[j]
                if not m.any():
                    continue   # the sentence's new values are absent from the evidence (unsupported)
                top = int(over[t].argmax())
                d = {"null": float(null[t]), "value": float(over[t][m].sum()), "chance": float(m.mean() * over[t].sum()),
                     "top_value": float(m[top])}
                rows[(stale, forms[j])].append(d)
                if len(record["evidence"]) > 1:
                    by_passage = np.zeros(len(record["evidence"]))
                    np.add.at(by_passage, passages, over[t])
                    holds = np.zeros(len(record["evidence"]), dtype=bool)
                    holds[np.array(passages)[m]] = True
                    passage_hits[stale].append(float(holds[int(by_passage.argmax())]))
                if stale and len(examples) < args.examples and text.strip():
                    order = np.argsort(-over[t])[:3]
                    examples.append({"id": ids[a], "sentence": j, "token": text, "null": round(float(null[t]), 3),
                                     "top": [(tok.decode([example.input_ids[example.evidence_spans[0][0] + k]]),
                                              round(float(over[t][k]), 3), bool(m[k])) for k in order]})

    def mean(key, items):
        return float(np.mean([d[key] for d in items])) if items else None
    out = {"articles": len(articles), "head": args.head, "features": args.features, "groups": {}, "passage": {}}
    for stale in (True, False):
        items = [d for (s, _), ds in rows.items() if s == stale for d in ds]
        out["groups"]["stale" if stale else "clean"] = {
            "tokens": len(items), "null": mean("null", items), "value": mean("value", items),
            "chance": mean("chance", items), "top_value": mean("top_value", items)}
        for f in ("year", "number", "name"):
            items = rows.get((stale, f), [])
            out["groups"][f"{'stale' if stale else 'clean'}_{f}"] = {
                "tokens": len(items), "null": mean("null", items), "value": mean("value", items),
                "chance": mean("chance", items), "top_value": mean("top_value", items)}
        hits = passage_hits.get(stale, [])
        out["passage"]["stale" if stale else "clean"] = {"tokens": len(hits), "top_passage": float(np.mean(hits)) if hits else None}
    out["examples"] = examples
    Path(args.out).write_text(json.dumps(out, indent=2))
    for name, g in out["groups"].items():
        if g["tokens"]:
            print(f"{name:14s} tokens {g['tokens']:6d}  null {g['null']:.3f}  value {g['value']:.3f}  "
                  f"chance {g['chance']:.3f}  top_value {g['top_value']:.3f}")
    for name, g in out["passage"].items():
        if g["tokens"]:
            print(f"passage {name:8s} tokens {g['tokens']:6d}  top_passage {g['top_passage']:.3f}")


if __name__ == "__main__":
    main()
