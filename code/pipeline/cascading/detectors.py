"""Three cascade detectors — Section 7.3.

After infilling round r, each detector flags sentences outside Δ_r that may
have been broken by the repairs. Flagged sentences enter the next round as
edit sentences.

Detectors (in increasing order of sophistication):

  (a) MentionOverlapDetector  — no training; flags by shared entity strings
  (b) StalenessDeltaDetector  — no training; flags by staleness score change
  (c) ConsequenceDetector     — trained head; Equation 8 of the paper

All three implement the same interface:

    detector.flag(
        sentences,          # list[str]  current article sentences
        delta_r,            # set[int]   indices repaired/deleted this round
        pooled_before,      # np.ndarray [n_sents, d] LLaDA states before repairs
        pooled_after,       # np.ndarray [n_sents, d] LLaDA states after repairs
        **kwargs,
    ) -> set[int]           # indices of flagged sentences (outside delta_r)

Usage:
    from cascading.detectors import MentionOverlapDetector, ConsequenceDetector
    det = MentionOverlapDetector()
    flagged = det.flag(sentences, delta_r={2}, pooled_before=H0, pooled_after=H1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cascading.mentions import Extractor, get_extractor

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# (a) Mention Overlap Detector
# ---------------------------------------------------------------------------

class MentionOverlapDetector:
    """Flag sentences that share a mention string with any repaired sentence.

    Paper §7.3(a): "Flag s_k when M(s_k) shares at least one string with the
    mentions of the sentences repaired this round."

    No training required. Expected: high recall, low precision.
    """

    def __init__(self, extractor: Extractor | None = None):
        self.M = extractor or get_extractor()

    def flag(
        self,
        sentences: list[str],
        delta_r: set[int],
        pooled_before: np.ndarray | None = None,  # unused, kept for interface parity
        pooled_after: np.ndarray | None = None,
        **kwargs,
    ) -> set[int]:
        """Return indices of sentences outside delta_r that share a mention with delta_r."""
        if not delta_r:
            return set()

        # Union of mentions from all repaired/deleted sentences
        repaired_mentions: set[str] = set()
        for j in delta_r:
            repaired_mentions |= self.M(sentences[j])

        if not repaired_mentions:
            return set()

        flagged: set[int] = set()
        for k, sent in enumerate(sentences):
            if k in delta_r:
                continue
            if self.M(sent) & repaired_mentions:
                flagged.add(k)
        return flagged


# ---------------------------------------------------------------------------
# (b) Staleness Delta Detector
# ---------------------------------------------------------------------------

class StalenessDeltaDetector:
    """Flag sentences whose tokens became more suspicious after repairs.

    Paper §7.3(b): Re-score every token under fresh hidden states. Flag s_k
    when one of its tokens (i) now scores ≥ τ_2, (ii) moved into the top 10th
    percentile of all article scores, and (iii) was not already there before.

    Ranks are compared (not raw score differences) to avoid artifacts from
    global sigmoid shifts between two encodings.

    Requires the staleness head and per-token pooled hidden states. The hidden
    states passed here are sentence-level pooled vectors; token-level rescoring
    would need the full token matrix. As a practical proxy this implementation
    scores each sentence's pooled vector and uses that as the sentence's scalar
    score, which approximates the per-token procedure over short sentences.
    """

    def __init__(
        self,
        staleness_head,                 # torch.nn.Linear (from staleness.train)
        mean: "torch.Tensor",           # standardisation mean
        std: "torch.Tensor",            # standardisation std
        tau2: float = 0.5,
        top_percentile: float = 90.0,
        extractor: Extractor | None = None,
    ):
        self.head = staleness_head
        self.mean = mean
        self.std = std
        self.tau2 = tau2
        self.top_percentile = top_percentile
        self.M = extractor or get_extractor()

    def _score_sentences(self, pooled: np.ndarray) -> np.ndarray:
        """Compute per-sentence staleness scores using the linear head."""
        import torch
        x = torch.from_numpy(pooled).float()
        with torch.inference_mode():
            scores = self.head((x - self.mean.cpu()) / self.std.cpu()).sigmoid().squeeze(-1)
        return scores.numpy()

    def flag(
        self,
        sentences: list[str],
        delta_r: set[int],
        pooled_before: np.ndarray,
        pooled_after: np.ndarray,
        **kwargs,
    ) -> set[int]:
        if pooled_before is None or pooled_after is None:
            raise ValueError("StalenessDeltaDetector requires pooled_before and pooled_after")

        scores_before = self._score_sentences(pooled_before)
        scores_after = self._score_sentences(pooled_after)

        # Rank of each sentence in its array (ascending: rank 0 = lowest score)
        rank_before = scores_before.argsort().argsort()
        rank_after = scores_after.argsort().argsort()

        top_threshold = np.percentile(scores_after, self.top_percentile)
        n = len(sentences)

        flagged: set[int] = set()
        for k in range(n):
            if k in delta_r:
                continue
            in_top_after = scores_after[k] >= top_threshold
            rose_in_rank = int(rank_after[k]) > int(rank_before[k])
            above_tau2 = scores_after[k] >= self.tau2
            if above_tau2 and in_top_after and rose_in_rank:
                flagged.add(k)
        return flagged


# ---------------------------------------------------------------------------
# (c) Consequence Prediction Detector
# ---------------------------------------------------------------------------

class ConsequenceDetector:
    """Flag sentences using the trained consequence prediction head (Eq. 8).

    For each repaired sentence s_j and each unrepaired sentence s_k, computes:
        g_jk = σ( MLP_ψ( [u_j ; u_k ; u_j ⊙ u_k ; |u_j − u_k|] ) )
    and flags s_k if g_jk ≥ τ_c for any j ∈ Δ_r.

    u_j comes from pooled_after (j already corrected).
    u_k comes from pooled_before (k not yet repaired).

    This detector can catch paraphrase cascades — dependence expressed without
    any shared surface mention string.
    """

    def __init__(self, head, tau_c: float = 0.5):
        """
        Args:
            head: cascading.train.ConsequenceHead (torch.nn.Module) in eval mode.
            tau_c: flagging threshold (tuned on validation).
        """
        self.head = head
        self.tau_c = tau_c

    def flag(
        self,
        sentences: list[str],
        delta_r: set[int],
        pooled_before: np.ndarray,
        pooled_after: np.ndarray,
        **kwargs,
    ) -> set[int]:
        import torch

        if not delta_r:
            return set()

        n = len(sentences)
        candidates = [k for k in range(n) if k not in delta_r]
        if not candidates:
            return set()

        H_before = torch.from_numpy(pooled_before).float()
        H_after = torch.from_numpy(pooled_after).float()

        flagged: set[int] = set()
        with torch.inference_mode():
            for j in delta_r:
                u_j = H_after[j].unsqueeze(0)                 # [1, d] corrected version
                u_k_batch = H_before[candidates]              # [n_cands, d] outdated versions
                # Broadcast u_j across candidates
                u_j_exp = u_j.expand(len(candidates), -1)
                scores = self.head(u_j_exp, u_k_batch).squeeze(-1)  # [n_cands]
                for idx, k in enumerate(candidates):
                    if scores[idx].item() >= self.tau_c:
                        flagged.add(k)
        return flagged


# ---------------------------------------------------------------------------
# Factory / loader
# ---------------------------------------------------------------------------

def load_consequence_head(path: str | Path, device: str = "cpu"):
    """Load a trained ConsequenceHead from a checkpoint directory.

    Returns: (head, config_dict)
    """
    import torch
    from cascading.train import ConsequenceHead

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    ckpt = torch.load(path / "head.pt", map_location=device)
    head = ConsequenceHead(d=config["d"], hidden=config["hidden"]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head, config


def build_detector(name: str, **kwargs):
    """Convenience factory.

    Args:
        name: one of 'mention', 'staleness_delta', 'consequence'
        **kwargs: passed to the detector constructor
    """
    if name == "mention":
        return MentionOverlapDetector(**kwargs)
    elif name == "staleness_delta":
        return StalenessDeltaDetector(**kwargs)
    elif name == "consequence":
        return ConsequenceDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector: {name!r}. Choose from: mention, staleness_delta, consequence")
