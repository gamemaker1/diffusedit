"""Article state, masking policy and candidate lengths (Sections 7.1 and 7.2).

The article is kept as token ids, one array per sentence, behind a fixed
prefix of [BOS] + evidence + separator. Nothing is re-tokenized during a run.
Text is decoded only for the verifier and the output. A deleted sentence
becomes an empty array, so sentence indices stay stable across rounds.

Every token carries a remask count. A span's replacement tokens take the
highest count in the span plus one, and a token at C_max is never masked
again (Section 7.4).

Spans are half-open (start, end) ranges local to their sentence.
"""

import math
from dataclasses import dataclass

import numpy as np

from infilling.model import MASK_ID

FALLBACK = 0.2  # empty mask set: mask this top fraction of the sentence
CAP = 0.5  # the mask set never exceeds this fraction of the sentence

RULES = {
    "zero": lambda n: 0,
    "half": lambda n: math.ceil(n / 2),
    "same": lambda n: n,
    "plus2": lambda n: n + 2,
    "double": lambda n: 2 * n,
}
LENGTH_SETS = {
    "same": ("same",),
    "keep-or-drop": ("zero", "same"),
    "full": ("zero", "half", "same", "plus2", "double"),
}


def lengths(n, name):
    """Candidate lengths Lambda(n) of Eq. 10 for a span of n masks."""
    return sorted({RULES[rule](n) for rule in LENGTH_SETS[name]})


@dataclass
class Article:
    bos: np.ndarray
    prefix: np.ndarray
    sentences: list[np.ndarray]
    counts: list[np.ndarray]

    @classmethod
    def from_example(cls, example, bos_id):
        ids = example.input_ids.astype(np.int64)
        first = example.spans[0][0]
        bos = ids[:1] if bos_id is not None and first and ids[0] == bos_id else ids[:0]
        sentences = [ids[a:b].copy() for a, b in example.spans]
        return cls(bos, ids[:first].copy(), sentences, [np.zeros(len(s), dtype=np.int64) for s in sentences])

    def offset(self, j):
        return len(self.prefix) + sum(len(s) for s in self.sentences[:j])

    def sequence(self, j=None, tokens=None):
        """The full input, with sentence j replaced by `tokens` if given."""
        parts = [self.prefix]
        for k, sentence in enumerate(self.sentences):
            parts.append(tokens if k == j else sentence)
        return np.concatenate(parts)

    def local(self, j, tokens):
        """[BOS] + neighbouring sentences only, for --pll-context local.

        Returns the sequence and the offset of `tokens` inside it.
        """
        before = self.sentences[j - 1] if j > 0 else self.prefix[:0]
        after = self.sentences[j + 1] if j + 1 < len(self.sentences) else self.prefix[:0]
        return np.concatenate([self.bos, before, tokens, after]), len(self.bos) + len(before)

    def drop(self, j):
        self.sentences[j] = self.sentences[j][:0]
        self.counts[j] = self.counts[j][:0]


def splice(array, spans, pieces):
    """Replace each span of `array` by the matching piece.

    Returns the new array and where each piece landed in it.
    """
    parts, placed, cursor, position = [], [], 0, 0
    for (a, b), piece in zip(spans, pieces):
        parts.append(array[cursor:a])
        position += a - cursor
        parts.append(np.asarray(piece, dtype=array.dtype))
        placed.append((position, position + len(piece)))
        position += len(piece)
        cursor = b
    parts.append(array[cursor:])
    return np.concatenate(parts), placed


def masks(n):
    return np.full(n, MASK_ID, dtype=np.int64)


def runs(mask):
    """Maximal runs of True as (start, end) spans."""
    spans, start = [], None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def oracle_spans(stale, counts, c_max):
    """Exactly the derived stale tokens, merged into runs (oracle token masks)."""
    return runs(np.asarray(stale, dtype=bool) & (counts < c_max))


def policy_spans(scores, counts, tau, c_max, stats):
    """The masking policy of Section 7.1 for one EDIT sentence.

    Tokens with score >= tau are masked. An empty set falls back to the top
    fifth of the sentence, a set over half the sentence keeps its top half,
    and one unmasked token between two masked ones joins them. Tokens at
    C_max are never masked. `stats` counts how often each rule fires.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    allowed = counts < c_max
    if not allowed.any():
        stats["exhausted"] += 1
        return []
    ranked = np.where(allowed, scores, -np.inf)
    mask = (scores >= tau) & allowed
    if not mask.any():
        mask[np.argsort(-ranked)[:min(math.ceil(FALLBACK * n), int(allowed.sum()))]] = True
        stats["fallback"] += 1
    limit = max(1, int(CAP * n))
    if mask.sum() > limit:
        keep = np.argsort(-np.where(mask, scores, -np.inf))[:limit]
        mask[:] = False
        mask[keep] = True
        stats["cap"] += 1
    for i in range(1, n - 1):
        if not mask[i] and mask[i - 1] and mask[i + 1] and allowed[i]:
            mask[i] = True
            stats["absorbed"] += 1
    return runs(mask)
