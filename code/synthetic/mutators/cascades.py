"""Cascading edit mutator for coupled factual dependencies (s_j -> s_k).

Creates scenarios where sentence j is a direct edit contradicted by evidence,
and sentence k is a downstream consequence (cascading edit) that contradicts
nothing in the evidence directly, but depends on the fact updated in sentence j.
"""

from dataclasses import dataclass
import random
import re

from .entities import ENTITY_POOLS, EntityMutator
from .numbers import NumberMutator


@dataclass
class CascadePair:
    direct_sentence_idx: int
    cascade_sentence_idx: int
    direct_original: str
    direct_mutated: str
    direct_stale_spans: list[tuple[int, int]]
    cascade_original: str
    cascade_mutated: str
    cascade_stale_spans: list[tuple[int, int]]
    shared_concept: str
    evidence_text: str


# Coupled facts: (entity_1, related_concept_1, entity_2, related_concept_2)
COUPLED_FACTS = [
    # Location and regional attribute
    ("Munich", "Bavarian", "Berlin", "Prussian"),
    ("Osaka", "Kansai", "Tokyo", "Kanto"),
    ("Toronto", "Ontario", "Montreal", "Quebec"),
    ("Sydney", "New South Wales", "Melbourne", "Victoria"),
    # Scale and classification
    ("50", "small workshop", "5000", "large enterprise"),
    ("3", "minor accident", "42", "major catastrophe"),
    ("12", "compact squad", "120", "full regiment"),
]


class CascadeMutator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.entity_mutator = EntityMutator(seed=seed)
        self.number_mutator = NumberMutator(seed=seed)

    def find_and_inject_cascade(
        self, sentences: list[str]
    ) -> CascadePair | None:
        """Find or inject a cascading dependency across two sentences in an article.

        Sentence j receives a direct factual error backed by evidence.
        Sentence k receives a downstream error that has NO direct evidence.
        """
        if len(sentences) < 2:
            return None

        # Strategy 1: Shared named entity across multiple sentences
        for j in range(len(sentences)):
            sent_j = sentences[j].strip()
            mut_j = self.entity_mutator.mutate(sent_j)
            if mut_j is None:
                continue

            orig_ent = mut_j.original_entity
            # Look for another sentence mentioning the same entity (with word boundary)
            pattern = re.compile(rf"\b{re.escape(orig_ent)}\b", re.IGNORECASE)
            for k in range(len(sentences)):
                if j == k:
                    continue
                sent_k = sentences[k].strip()
                match = pattern.search(sent_k)
                if match:
                    start, end = match.span()
                    mutated_k = sent_k[:start] + mut_j.mutated_entity + sent_k[end:]
                    stale_span_k = (start, start + len(mut_j.mutated_entity))

                    evidence_text = f"According to official records, {orig_ent} was verified in {sent_j}."

                    return CascadePair(
                        direct_sentence_idx=j,
                        cascade_sentence_idx=k,
                        direct_original=sent_j,
                        direct_mutated=mut_j.mutated_text,
                        direct_stale_spans=mut_j.stale_spans,
                        cascade_original=sent_k,
                        cascade_mutated=mutated_k,
                        cascade_stale_spans=[stale_span_k],
                        shared_concept=orig_ent,
                        evidence_text=evidence_text,
                    )

        # Strategy 2: Pre-defined coupled semantic concepts
        indices = list(range(len(sentences)))
        self.rng.shuffle(indices)
        j, k = indices[0], indices[1]
        sent_j, sent_k = sentences[j].strip(), sentences[k].strip()

        # Check if any coupled concept appears in sent_j or sent_k
        for (f1_a, f1_b, f2_a, f2_b) in COUPLED_FACTS:
            if f1_a in sent_j:
                mut_j_text = sent_j.replace(f1_a, f2_a)
                start_j = mut_j_text.find(f2_a)
                span_j = (start_j, start_j + len(f2_a))

                # Inject related concept into sent_k if applicable
                mut_k_text = sent_k + f" As a result, the operation was classified as a {f2_b}."
                orig_k_text = sent_k + f" As a result, the operation was classified as a {f1_b}."
                start_k = mut_k_text.rfind(f2_b)
                span_k = (start_k, start_k + len(f2_b))

                evidence_text = f"Records establish {f1_a} as the confirmed setting."
                return CascadePair(
                    direct_sentence_idx=j,
                    cascade_sentence_idx=k,
                    direct_original=sent_j,
                    direct_mutated=mut_j_text,
                    direct_stale_spans=[span_j],
                    cascade_original=orig_k_text,
                    cascade_mutated=mut_k_text,
                    cascade_stale_spans=[span_k],
                    shared_concept=f"{f1_a} -> {f1_b}",
                    evidence_text=evidence_text,
                )

        return None
