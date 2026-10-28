"""Four cascade detectors — Section 7.3 and progress report.

After infilling round r, each detector flags sentences outside Δ_r that may
have been broken by the repairs. Flagged sentences enter the next round as
edit sentences.

Detectors, lettered as in the paper, with the progress report's addition last:

  (a) MentionOverlapDetector      — no training; flags by shared entity strings
  (b) StalenessDeltaDetector      — no training; flags by staleness score change
  (c) ConsequenceDetector         — trained head; Equation 8 of the paper
  (d) EditedAsEvidenceDetector    — adds edited sentences to the evidence for round r+1

All detectors implement the standard interface:

    detector.flag(
        sentences,          # list[str]  current article sentences
        delta_r,            # set[int]   indices repaired/deleted this round
        pooled_before,      # np.ndarray [n_sents, d] LLaDA states before repairs
        pooled_after,       # np.ndarray [n_sents, d] LLaDA states after repairs
        **kwargs,
    ) -> set[int]           # indices of flagged sentences (outside delta_r)

Usage:
    from cascading.detectors import (
        MentionOverlapDetector,
        EditedAsEvidenceDetector,
        StalenessDeltaDetector,
        ConsequenceDetector,
    )
    det = MentionOverlapDetector()
    flagged = det.flag(sentences, delta_r={0}, previous=sentences_before_round)
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
        previous: list[str] | None = None,
        **kwargs,
    ) -> set[int]:
        """Return indices of sentences outside delta_r that share a mention with delta_r.

        `previous` holds the sentences before this round's repairs. The mentions
        of a repaired sentence are those of both its versions, because a
        dependent sentence shares the old mention ("three"), not the new one,
        and a deleted sentence has no current text.
        """
        if not delta_r:
            return set()

        repaired_mentions: set[str] = set()
        for j in delta_r:
            repaired_mentions |= self.M(sentences[j])
            if previous is not None:
                repaired_mentions |= self.M(previous[j])

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
# (d) Using Edited Sentences as New Evidence Detector
# ---------------------------------------------------------------------------

class EditedAsEvidenceDetector:
    """Using edited sentences as new evidence for cascade detection.

    Paper / progress report:
    "If the evidence holds k snippets in round r and p sentences are edited in
    that round, we append the p edited sentences to the evidence and use the
    k + p snippets in round r + 1. A sentence that a repair made wrong then
    conflicts with a sentence in the evidence, so the existing heads can flag it."

    In this strategy, the cascade detector updates the evidence pool:
      E^{(r+1)} = E^{(r)} ∪ {edited sentences from round r}

    When round r+1 runs its standard forward pass over (E^{(r+1)} ⊕ A^{(r)}),
    the existing sentence classifier head naturally identifies any sentence
    that conflicts with the newly introduced evidence.
    """

    def __init__(self, title_prefix: str = "Updated Fact (Round {round})"):
        self.title_prefix = title_prefix

    def augment_evidence(
        self,
        evidence: list[dict],
        repaired_sentences: list[str] | dict[int, str],
        round_idx: int = 1,
    ) -> list[dict]:
        """Add the p edited sentences from round r to the evidence for round r + 1.

        The new snippets go before the old ones, because classifier.inputs cuts
        the evidence at its token budget from the end, and a cut must never
        remove the sentences this detector exists to add.

        Args:
            evidence: list of current snippet dicts [{"title": ..., "text": ...}, ...]
            repaired_sentences: the p sentences edited in round r (list or dict).
            round_idx: current round number r.

        Returns:
            Updated list with k + p snippets.
        """
        added = []
        if isinstance(repaired_sentences, dict):
            items = list(repaired_sentences.values())
        else:
            items = list(repaired_sentences)

        for p_idx, text in enumerate(items, 1):
            text_clean = text.strip()
            if not text_clean:
                continue
            title = self.title_prefix.format(round=round_idx)
            if len(items) > 1:
                title = f"{title} #{p_idx}"
            added.append({
                "title": title,
                "text": text_clean,
            })
        return added + list(evidence)

    def flag(
        self,
        sentences: list[str],
        delta_r: set[int],
        classifier_probs: np.ndarray | None = None,
        **kwargs,
    ) -> set[int]:
        """Return sentences outside delta_r whose round r+1 arg-max class is not KEEP.

        classifier_probs is [n_sents, 3], KEEP/EDIT/DROP, from the classifier
        over (E^{(r+1)} ⊕ A^{(r)}). A DROP arg-max that fails the gate tau_d
        becomes EDIT (Section 6.2), so both non-KEEP classes flag the sentence,
        and the gate decides between deletion and re-masking afterwards.
        """
        if not delta_r:
            return set()
        if classifier_probs is None:
            raise ValueError("EditedAsEvidenceDetector requires classifier_probs")
        top = np.asarray(classifier_probs).argmax(-1)
        return {k for k in range(len(sentences)) if k not in delta_r and top[k] != 0}

# ---------------------------------------------------------------------------
# (b) Staleness Delta Detector
# ---------------------------------------------------------------------------

class StalenessDeltaDetector:
    """Flag sentences whose tokens became more suspicious after the repairs (Section 7.3(b)).

    With token_scores_before and token_scores_after, the staleness head's
    scores of every token of every sentence under the states before and
    after the round, s_k is flagged when one of its tokens now scores at
    least tau_2 and lies in the top tenth of the article's token scores,
    where it was not before the repairs. This is the rule of the paper, and
    infilling.loop uses it. The tokens of a sentence outside Delta_r are
    unchanged, so token i of s_k is the same token in both encodings.

    With pooled_before and pooled_after only, as in the offline evaluation
    of cascading.evaluate, which has no token states for every sentence, the
    same rule runs on one score per sentence from its pooled vector. Deleted
    sentences, with no tokens, take no part in either ranking.
    """

    def __init__(
        self,
        staleness_head=None,            # torch.nn.Linear from staleness.train, for the pooled rule
        mean: "torch.Tensor" = None,
        std: "torch.Tensor" = None,
        tau2: float = 0.5,
        top_percentile: float = 90.0,
    ):
        self.head = staleness_head
        self.mean = mean
        self.std = std
        self.tau2 = tau2
        self.top_percentile = top_percentile

    def _score_sentences(self, pooled: np.ndarray) -> np.ndarray:
        import torch
        x = torch.from_numpy(pooled).float()
        with torch.inference_mode():
            scores = self.head((x - self.mean.cpu()) / self.std.cpu()).sigmoid().squeeze(-1)
        return scores.numpy()

    def _rising(self, before: list[np.ndarray], after: list[np.ndarray]) -> list[np.ndarray]:
        """Per sentence, a boolean array over its units: in the top tenth after, not before, and >= tau_2."""
        live = [k for k, (x, y) in enumerate(zip(before, after)) if len(x) and len(y)]
        if not live:
            return [np.zeros(len(y), dtype=bool) for y in after]
        cut_before = np.percentile(np.concatenate([before[k] for k in live]), self.top_percentile)
        cut_after = np.percentile(np.concatenate([after[k] for k in live]), self.top_percentile)
        out = []
        for k, (x, y) in enumerate(zip(before, after)):
            if k not in live or len(x) != len(y):
                out.append(np.zeros(len(y), dtype=bool))
                continue
            out.append((y >= cut_after) & ~(x >= cut_before) & (y >= self.tau2))
        return out

    def flag(
        self,
        sentences: list[str],
        delta_r: set[int],
        pooled_before: np.ndarray | None = None,
        pooled_after: np.ndarray | None = None,
        token_scores_before: list[np.ndarray] | None = None,
        token_scores_after: list[np.ndarray] | None = None,
        **kwargs,
    ) -> set[int]:
        if token_scores_before is not None and token_scores_after is not None:
            rising = self._rising(token_scores_before, token_scores_after)
        elif pooled_before is not None and pooled_after is not None and self.head is not None:
            empty = {k for k, s in enumerate(sentences) if not s.strip()}
            before, after = self._score_sentences(pooled_before), self._score_sentences(pooled_after)
            rising = self._rising([np.array([] if k in empty else [v]) for k, v in enumerate(before)],
                                  [np.array([] if k in empty else [v]) for k, v in enumerate(after)])
        else:
            raise ValueError("StalenessDeltaDetector needs token scores, or pooled states and a head")
        return {k for k, r in enumerate(rising) if k not in delta_r and r.any()}


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
                scores = self.head(u_j_exp, u_k_batch).squeeze(-1).sigmoid()  # [n_cands], g_jk
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
        name: one of 'mention', 'edited_as_evidence', 'evidence', 'staleness_delta', 'consequence'
        **kwargs: passed to the detector constructor
    """
    if name == "mention":
        return MentionOverlapDetector(**kwargs)
    elif name in ("edited_as_evidence", "evidence"):
        return EditedAsEvidenceDetector(**kwargs)
    elif name == "staleness_delta":
        return StalenessDeltaDetector(**kwargs)
    elif name == "consequence":
        return ConsequenceDetector(**kwargs)
    else:
        raise ValueError(
            f"Unknown detector: {name!r}. Choose from: mention, edited_as_evidence, staleness_delta, consequence"
        )

