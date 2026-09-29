"""Candidate reranking (Section 7.2, Eqs. 11 and 12).

Every candidate is a whole sentence with its spans filled. Fluency is the
masked pseudo-log-likelihood, where each probed token is masked alone and
scored by the frozen model. Support is the VitaminC verifier's P(SUPPORTS)
against each evidence snippet, keeping the best snippet. The score is

    pll       R = PLL
    verifier  R = Ent
    combined  R = lam * PLL + (1 - lam) * Ent

A deletion candidate (every span at length 0) wins only when its score beats
the best rewrite by the deletion margin, since removing a span also removes
the claim the verifier would have checked.

Two options trade exactness for cost and are off by default, matching the
paper. --pll-context local scores against the neighbouring sentences instead
of the full evidence + article input. --pll-window w probes only the tokens
within w of a span instead of the whole sentence. --pll-norm exp maps the
average log-probability to a geometric-mean probability in (0, 1], the same
range as Ent, so that lam weighs comparable quantities.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from infilling.model import MASK_ID

VERIFIER = "tals/albert-base-vitaminc-mnli"


@dataclass
class Candidate:
    tokens: np.ndarray
    lengths: tuple[int, ...]
    windows: list[tuple[int, int]]
    deletion: bool
    confidence: list[float] = field(default_factory=list)
    duplicate_of: int | None = None
    pll: float | None = None
    ent: float | None = None
    score: float | None = None


def probe_positions(n, windows, window):
    if not window:
        return list(range(n))
    return sorted({i for a, b in windows for i in range(max(0, a - window), min(n, b + window))})


def pll(model, article, j, candidate, settings):
    """Average log-probability of the probed tokens, each masked alone (Eq. 11)."""
    tokens = candidate.tokens
    if settings.pll_context == "local":
        seq, offset = article.local(j, tokens)
    else:
        seq, offset = article.sequence(j, tokens), article.offset(j)
    probes = probe_positions(len(tokens), candidate.windows, settings.pll_window)
    if not probes:
        probes = list(range(len(tokens)))
    seq = torch.from_numpy(seq).to(model.device)
    total = 0.0
    for i in range(0, len(probes), settings.batch):
        chunk = torch.tensor(probes[i:i + settings.batch], device=model.device) + offset
        ids = seq.repeat(len(chunk), 1)
        rows = torch.arange(len(chunk), device=model.device)
        true = ids[rows, chunk]
        ids[rows, chunk] = MASK_ID
        logp = model.logits_at(ids, chunk[:, None], "pll")[:, 0].log_softmax(-1)
        total += logp[rows, true].sum().item()
    mean = total / len(probes)
    return float(np.exp(mean)) if settings.pll_norm == "exp" else mean


class Verifier:
    """VitaminC fact verifier, P(SUPPORTS | evidence, claim)."""

    def __init__(self, name, device, support_index=None, batch=32):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.net = AutoModelForSequenceClassification.from_pretrained(name).eval().to(device)
        self.device, self.batch = device, batch
        if support_index is None:
            names = {i: str(label).lower() for i, label in self.net.config.id2label.items()}
            found = [i for i, label in names.items() if "support" in label or "entail" in label]
            if len(found) != 1:
                raise SystemExit(f"{name}: cannot tell the SUPPORTS label from {names}, pass --support-index")
            support_index = found[0]
        self.support = int(support_index)
        self.pairs = 0

    @torch.inference_mode()
    def __call__(self, claims, evidence):
        """Per claim, the highest P(SUPPORTS) over the evidence snippets."""
        if not evidence:
            return [0.0] * len(claims)
        pairs = [(c, e) for c in claims for e in evidence]
        self.pairs += len(pairs)
        probs = []
        for i in range(0, len(pairs), self.batch):
            chunk = pairs[i:i + self.batch]
            enc = self.tok([c for c, _ in chunk], [e for _, e in chunk], truncation=True,
                           max_length=256, padding=True, return_tensors="pt").to(self.device)
            probs.append(self.net(**enc).logits.float().softmax(-1)[:, self.support].cpu())
        return torch.cat(probs).view(len(claims), len(evidence)).max(1).values.tolist()


def score(model, verifier, article, j, candidates, evidence, tok, settings):
    """Fill pll, ent and score on every candidate. Duplicates copy their original."""
    unique = [c for c in candidates if c.duplicate_of is None]
    if settings.scorer in ("pll", "combined"):
        for c in unique:
            c.pll = pll(model, article, j, c, settings)
    if settings.scorer in ("verifier", "combined"):
        texts = [tok.decode(c.tokens.tolist()).strip() for c in unique]
        ents = verifier(texts, evidence) if verifier is not None else [0.0] * len(unique)
        for c, ent in zip(unique, ents):
            c.ent = ent
    for c in candidates:
        if c.duplicate_of is not None:
            original = candidates[c.duplicate_of]
            c.pll, c.ent = original.pll, original.ent
        if settings.scorer == "pll":
            c.score = c.pll
        elif settings.scorer == "verifier":
            c.score = c.ent
        else:
            c.score = settings.lam * c.pll + (1 - settings.lam) * c.ent


def pick(candidates, margin):
    """Index of the winner, applying the deletion margin."""
    rewrites = [i for i, c in enumerate(candidates) if not c.deletion]
    deletions = [i for i, c in enumerate(candidates) if c.deletion]
    best = max(rewrites, key=lambda i: candidates[i].score) if rewrites else None
    if deletions:
        d = deletions[0]
        if best is None or candidates[d].score > candidates[best].score + margin:
            return d
    return best
