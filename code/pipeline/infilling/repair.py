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

With keep_original, each choice also scores the span's original tokens as one
more candidate, so a span the staleness head masked by mistake can stay as it
was. It wins only when it outscores every rewrite.

With evidence_copy k (sequential only), each choice also scores k candidates
that put an evidence phrase into the span verbatim, as VENCE (Chen et al.
2022) proposes entities from the evidence. The phrases are the evidence's
named entities (spaCy, from infilling.ner) that the sentence lacks, of the
masked entity's type when the masked text is part of an entity, else of its
kind (names or numbers). Without an entity file they are the capitalised names
and the numbers of the evidence. The k kept are those whose tokens LLaDA
finds most likely in the masked span, one forward pass per phrase length.
Later spans are still sampled. A sampled fill can only hold words the model proposes,
while these candidates hold the evidence's words whatever the model prefers.

The article is updated in place, so later sentences see earlier repairs.
"""

import re
from dataclasses import dataclass

import numpy as np
import torch

from infilling.model import MASK_ID
from infilling.sampler import sample
from infilling.score import Candidate, adjust, pick, score
from infilling.spans import LENGTH_SETS, RULES, evidence_words, lengths, masks, splice, words

CONNECTORS = {"of", "de", "del", "da", "di", "du", "van", "von", "der", "den", "la", "le", "al", "el", "y"}
PIECE = re.compile(r"\d[\d,.]*\d(?:s|th|st|nd|rd)?|[\w'-]+|[^\w\s]")
MAX_PHRASE = 12
NUMERIC = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
NAMED = {"PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}
NUMBER_WORDS = set("""one two three four five six seven eight nine ten eleven twelve thirteen fourteen
fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred
thousand million billion first second third fourth fifth sixth seventh eighth ninth tenth dozen""".split())
FUNCTION_WORDS = {"the", "a", "an", "of", "in", "on", "at", "and", "for", "to", "by"}


def names_in(text):
    """Numbers, and runs of capitalised words that may hold connectors such as "of"."""
    out, run = [], []

    def flush():
        while run and run[-1].lower() in CONNECTORS:
            run.pop()
        if run:
            out.append(" ".join(run))
        run.clear()

    for w in PIECE.findall(text):
        if w[0].isdigit():
            flush()
            out.append(w)
        elif w[0].isupper() and w.endswith("'s"):
            run.append(w[:-2])
            flush()
        elif w[0].isupper():
            run.append(w)
        elif run and w.lower() in CONNECTORS:
            run.append(w)
        else:
            flush()
    flush()
    return out


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
    keep_original: bool = False
    edit_penalty: float = 0.0
    reject_unsupported: bool = False
    evidence_copy: int = 0


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


def candidate(model, article, j, tokens, spans, lens, decided, settings, generator, keep=(), fixed=None):
    """Candidate with spans[k] at length lens[k], its original tokens for k in `keep`,
    or the given tokens for k in `fixed`. PLL windows cover the spans being decided."""
    fixed = fixed or {}
    pieces = [fixed[k] if k in fixed else tokens[a:b] if k in keep else masks(n)
              for k, ((a, b), n) in enumerate(zip(spans, lens))]
    masked, placed = splice(tokens, spans, pieces)
    filled, confidence = fill(model, article, j, masked, settings, generator)
    return Candidate(filled, tuple(len(p) for p in pieces), [placed[k] for k in decided],
                     all(len(pieces[k]) == 0 for k in decided), confidence, original=bool(keep))


def kind_of(text):
    """NUMERIC or NAMED for text holding a number or a capitalised word, else None."""
    if any(ch.isdigit() for ch in text) or set(words(text)) & NUMBER_WORDS:
        return NUMERIC
    if any(ch.isupper() for ch in text):
        return NAMED
    return None


def phrases(evidence, masked_text, sentence_text, entities=None, j=None):
    """Evidence phrases the sentence lacks, of the masked text's entity type or kind."""
    have, out = sentence_text.lower(), []
    if entities is not None:
        masked = set(words(masked_text)) - FUNCTION_WORDS
        labels = {label for text, label in entities["sentences"][j] if masked & set(words(text))}
        allowed = labels or kind_of(masked_text) or set()
        found = [(t, label) for t, label in entities["evidence"] if label in allowed]
        if labels and not found:
            group = NUMERIC if labels & NUMERIC else NAMED
            found = [(t, label) for t, label in entities["evidence"] if label in group]
        for t, _ in found:
            if t.lower() not in have and t not in out:
                out.append(t)
        return out
    kind = kind_of(masked_text)
    if kind is None:
        return []
    for snippet in evidence:
        for p in names_in(snippet):
            if (p[0].isdigit()) == (kind is NUMERIC) and p.lower() not in have and p not in out:
                out.append(p)
    return out


@torch.inference_mode()
def shortlist(model, article, j, tokens, spans, later, options, lead, tok, k):
    """The k phrases whose tokens are most likely in span 0, by mean log-probability
    with the span masked at the phrase's token length and later spans masked."""
    ids = {p: tok.encode(lead + p, add_special_tokens=False) for p in options}
    by_length = {}
    for p, t in ids.items():
        if 0 < len(t) <= MAX_PHRASE:
            by_length.setdefault(len(t), []).append(p)
    scored = []
    for n, group in by_length.items():
        masked, placed = splice(tokens, spans, [masks(n)] + [masks(m) for m in later])
        seq = torch.from_numpy(article.sequence(j, masked)).to(model.device)[None]
        start = article.offset(j) + placed[0][0]
        positions = torch.arange(start, start + n, device=model.device)[None]
        logp = model.logits_at(seq, positions, "copy")[0].log_softmax(-1)
        rows = torch.arange(n, device=model.device)
        for p in group:
            scored.append((logp[rows, torch.tensor(ids[p], device=model.device)].mean().item(), p))
    return [(p, np.asarray(ids[p], dtype=tokens.dtype)) for _, p in sorted(scored, reverse=True)[:k]]


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


def decide(model, verifier, article, j, candidates, evidence, tok, settings, stats, reference=""):
    score(model, verifier, article, j, candidates, evidence, tok, settings)
    adjust(candidates, reference, evidence_words(evidence), tok, settings, stats)
    chosen = pick(candidates, settings.deletion_margin)
    if any(c.deletion for c in candidates):
        stats["deletion_candidates"] += 1
        stats["deletion_wins"] += candidates[chosen].deletion
    if any(c.original for c in candidates):
        stats["keep_candidates"] += 1
        stats["keep_wins"] += candidates[chosen].original
    if any(c.copied for c in candidates):
        stats["copy_candidates"] += 1
        stats["copy_wins"] += candidates[chosen].copied is not None
    return chosen


def record(candidates, chosen, tok):
    return {
        "chosen": chosen,
        "candidates": [{
            "lengths": list(c.lengths),
            "text": tok.decode(c.tokens.tolist()).strip(),
            "deletion": c.deletion,
            "original": c.original,
            "copied": c.copied,
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


def repair_sentence(model, verifier, article, j, spans, evidence, tok, settings, generator, stats, entities=None):
    """Repair sentence j at `spans` in place. Returns one decision record per choice made.
    `entities` is the article's entry of infilling.ner, for evidence_copy."""
    spans = sorted(spans)
    stats["spans"] += len(spans)
    tokens, counts = article.sentences[j], article.counts[j]
    reference = tok.decode(tokens.tolist())
    # Later spans hold sampled tokens once an earlier span is decided, so the
    # evidence-copy step reads each span's text from the sentence before repair.
    originals = [tok.decode(tokens[a:b].tolist()) for a, b in spans]
    decisions = []

    if settings.multi_span == "shared-ratio":
        seen, candidates = set(), []
        for rule in LENGTH_SETS[settings.lengths]:
            lens = tuple(RULES[rule](b - a) for a, b in spans)
            if lens not in seen:
                seen.add(lens)
                candidates.append(candidate(model, article, j, tokens, spans, lens,
                                            range(len(spans)), settings, generator))
        if settings.keep_original:
            candidates.append(candidate(model, article, j, tokens, spans, [b - a for a, b in spans],
                                        range(len(spans)), settings, generator, keep=range(len(spans))))
        candidates = dedupe(candidates, stats)
        chosen = decide(model, verifier, article, j, candidates, evidence, tok, settings, stats, reference)
        counts = bump(counts, spans, candidates[chosen].lengths)
        tokens = candidates[chosen].tokens
        decisions.append(record(candidates, chosen, tok))
    else:
        for i in range(len(spans)):
            a, b = spans[i]
            later = [y - x for x, y in spans[i + 1:]]
            candidates = [candidate(model, article, j, tokens, spans[i:], [n] + later, [0], settings, generator)
                          for n in lengths(b - a, settings.lengths)]
            if settings.keep_original:
                candidates.append(candidate(model, article, j, tokens, spans[i:], [b - a] + later, [0],
                                            settings, generator, keep={0}))
            if settings.evidence_copy:
                masked_text = originals[i]
                options = phrases(evidence, masked_text, reference, entities, j)
                lead = " " if masked_text[:1].isspace() else ""
                for phrase, ids in shortlist(model, article, j, tokens, spans[i:], later, options, lead,
                                             tok, settings.evidence_copy):
                    c = candidate(model, article, j, tokens, spans[i:], [len(ids)] + later, [0],
                                  settings, generator, fixed={0: ids})
                    c.copied = phrase
                    candidates.append(c)
            candidates = dedupe(candidates, stats)
            chosen = decide(model, verifier, article, j, candidates, evidence, tok, settings, stats, reference)
            n = candidates[chosen].lengths[0]
            counts = bump(counts, [spans[i]], [n])
            tokens = candidates[chosen].tokens
            shift = n - (b - a)
            spans[i + 1:] = [(x + shift, y + shift) for x, y in spans[i + 1:]]
            decisions.append({"span": i, **record(candidates, chosen, tok)})

    article.sentences[j], article.counts[j] = tokens, counts
    return decisions
