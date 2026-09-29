"""Repair one EDIT sentence: infill its masked spans at several lengths and
keep the best-scoring candidate (Section 7.2, Algorithm 1 line 5).

A sentence with k spans cannot try every combination of lengths, which is
|Lambda|^k candidates. --multi-span picks one of two strategies.

    sequential    Spans are decided left to right. For span i, each candidate
                  length is sampled together with the later spans, masked at
                  their own length, so every candidate is a whole sentence.
                  The winner is committed and span i+1 is decided next.
                  Costs |Lambda| sampler runs per span.
    shared-ratio  Every span uses the same length rule (all half, all same,
                  ...), giving |Lambda| candidates per sentence in total.

The article is updated in place, so later sentences see earlier repairs.
"""

from dataclasses import dataclass

import numpy as np
import torch

from infilling.model import MASK_ID
from infilling.sampler import sample
from infilling.score import Candidate, pick, score
from infilling.spans import LENGTH_SETS, RULES, lengths, masks, splice


@dataclass
class Settings:
    lengths: str = "full"
    multi_span: str = "sequential"
    steps: int = 8
    temperature: float = 0.0
    scorer: str = "combined"
    lam: float = 0.5
    deletion_margin: float = 0.1
    pll_context: str = "full"
    pll_window: int = 0
    pll_norm: str = "raw"
    batch: int = 8


def fill(model, article, j, masked, settings, generator):
    """Sample every mask in sentence j's `masked` tokens, in full-article context."""
    local = np.flatnonzero(masked == MASK_ID)
    if not len(local):
        return masked, []
    offset = article.offset(j)
    ids = torch.from_numpy(article.sequence(j, masked)).to(model.device)
    positions = torch.from_numpy(local + offset).to(model.device)
    out, confidence = sample(model, ids, positions, settings.steps, settings.temperature, generator)
    return out[offset:offset + len(masked)].cpu().numpy(), confidence.tolist()


def candidate(model, article, j, tokens, spans, lens, decided, settings, generator):
    """Candidate with spans[k] at length lens[k]. PLL windows cover the spans being decided."""
    masked, placed = splice(tokens, spans, [masks(n) for n in lens])
    filled, confidence = fill(model, article, j, masked, settings, generator)
    return Candidate(filled, tuple(lens), [placed[k] for k in decided],
                     all(lens[k] == 0 for k in decided), confidence)


def dedupe(candidates, stats):
    kept = []
    for i, c in enumerate(candidates):
        if not len(c.tokens):
            stats["empty"] += 1
            continue
        for k, other in enumerate(kept):
            if other.duplicate_of is None and np.array_equal(other.tokens, c.tokens):
                c.duplicate_of = k
                stats["duplicates"] += 1
                break
        kept.append(c)
    return kept


def decide(model, verifier, article, j, candidates, evidence, tok, settings, stats):
    score(model, verifier, article, j, candidates, evidence, tok, settings)
    chosen = pick(candidates, settings.deletion_margin)
    if any(c.deletion for c in candidates):
        stats["deletion_candidates"] += 1
        stats["deletion_wins"] += candidates[chosen].deletion
    return chosen


def record(candidates, chosen, tok):
    return {
        "chosen": chosen,
        "candidates": [{
            "lengths": list(c.lengths),
            "text": tok.decode(c.tokens.tolist()).strip(),
            "deletion": c.deletion,
            "duplicate_of": c.duplicate_of,
            "pll": c.pll,
            "ent": c.ent,
            "score": c.score,
            "confidence": c.confidence,
        } for c in candidates],
    }


def bump(counts, spans, lens):
    """Replacement tokens take the span's highest remask count plus one."""
    pieces = [np.full(n, (counts[a:b].max() if b > a else 0) + 1, dtype=counts.dtype)
              for (a, b), n in zip(spans, lens)]
    return splice(counts, spans, pieces)[0]


def repair_sentence(model, verifier, article, j, spans, evidence, tok, settings, generator, stats):
    """Repair sentence j at `spans` in place. Returns one decision record per choice made."""
    spans = sorted(spans)
    stats["spans"] += len(spans)
    tokens, counts = article.sentences[j], article.counts[j]
    decisions = []

    if settings.multi_span == "shared-ratio":
        seen, candidates = set(), []
        for rule in LENGTH_SETS[settings.lengths]:
            lens = tuple(RULES[rule](b - a) for a, b in spans)
            if lens not in seen:
                seen.add(lens)
                candidates.append(candidate(model, article, j, tokens, spans, lens,
                                            range(len(spans)), settings, generator))
        candidates = dedupe(candidates, stats)
        chosen = decide(model, verifier, article, j, candidates, evidence, tok, settings, stats)
        counts = bump(counts, spans, candidates[chosen].lengths)
        tokens = candidates[chosen].tokens
        decisions.append(record(candidates, chosen, tok))
    else:
        for i in range(len(spans)):
            a, b = spans[i]
            later = [y - x for x, y in spans[i + 1:]]
            candidates = dedupe([candidate(model, article, j, tokens, spans[i:], [n] + later,
                                           [0], settings, generator)
                                 for n in lengths(b - a, settings.lengths)], stats)
            chosen = decide(model, verifier, article, j, candidates, evidence, tok, settings, stats)
            n = candidates[chosen].lengths[0]
            counts = bump(counts, [spans[i]], [n])
            tokens = candidates[chosen].tokens
            shift = n - (b - a)
            spans[i + 1:] = [(x + shift, y + shift) for x, y in spans[i + 1:]]
            decisions.append({"span": i, **record(candidates, chosen, tok)})

    article.sentences[j], article.counts[j] = tokens, counts
    return decisions
