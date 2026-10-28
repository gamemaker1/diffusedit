"""Cascading edit mutator for coupled factual dependencies (s_j -> s_k).

Sentence j receives a direct error that the evidence corrects. Sentence k
receives an error that follows from the one in j, and no snippet mentions k.
Each strategy changes text the seed article already holds, so no template
sentence is added. The strategies are tried in random order.

    shared   j and k name the same entity, and both name the replacement.
             The cascade shares a mention with the direct edit.
    alias    j names a person in full ("James J. Hill") and k by surname
             alone ("Hill"). j names another person and k that person's
             surname. The mention strings differ.
    demonym  j names a country and k uses its demonym ("France", "French").
             j names another country and k its demonym.
"""

from dataclasses import dataclass
import random
import re

from .entities import EntityBank, EntityMutator, Replacement, apply

DEMONYMS = {
    "Argentina": "Argentine", "Australia": "Australian", "Austria": "Austrian", "Belgium": "Belgian",
    "Brazil": "Brazilian", "Bulgaria": "Bulgarian", "Canada": "Canadian", "Chile": "Chilean", "China": "Chinese",
    "Colombia": "Colombian", "Croatia": "Croatian", "Cuba": "Cuban", "Denmark": "Danish", "Egypt": "Egyptian",
    "England": "English", "Estonia": "Estonian", "Finland": "Finnish", "France": "French", "Germany": "German",
    "Ghana": "Ghanaian", "Greece": "Greek", "Hungary": "Hungarian", "Iceland": "Icelandic", "India": "Indian",
    "Indonesia": "Indonesian", "Iran": "Iranian", "Ireland": "Irish", "Israel": "Israeli", "Italy": "Italian",
    "Jamaica": "Jamaican", "Japan": "Japanese", "Kenya": "Kenyan", "Latvia": "Latvian", "Lithuania": "Lithuanian",
    "Mexico": "Mexican", "Morocco": "Moroccan", "Nigeria": "Nigerian", "Norway": "Norwegian", "Pakistan": "Pakistani",
    "Peru": "Peruvian", "Poland": "Polish", "Portugal": "Portuguese", "Romania": "Romanian", "Russia": "Russian",
    "Scotland": "Scottish", "Serbia": "Serbian", "Spain": "Spanish", "Sweden": "Swedish",
    "Switzerland": "Swiss", "Thailand": "Thai", "Turkey": "Turkish", "Ukraine": "Ukrainian", "Vietnam": "Vietnamese",
    "Wales": "Welsh",
}
STRATEGIES = ("shared", "alias", "demonym")


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
    strategy: str


def word(text):
    return re.compile(rf"(?<!\w){re.escape(text)}(?!\w)")


class CascadeMutator:
    def __init__(self, seed: int | None = None, bank: EntityBank | None = None):
        self.rng = random.Random(seed)
        self.bank = bank
        self.entity_mutator = EntityMutator(seed=seed, bank=bank)

    def _pair(self, sentences, j, k, rj, rks, concept, strategy):
        mutated_j, spans_j = apply(sentences[j], rj)
        mutated_k, spans_k = apply(sentences[k], rks)
        return CascadePair(j, k, sentences[j], mutated_j, spans_j, sentences[k], mutated_k, spans_k, concept, strategy)

    def _shared(self, sentences, article, order):
        for j in order:
            for start, end, entity, label in self.entity_mutator._extract_entities(sentences[j]):
                ks = [k for k in order if k != j and word(entity).search(sentences[k])]
                if not ks:
                    continue
                new = self.entity_mutator.replacement_for(entity, label, article)
                if not new:
                    continue
                k = self.rng.choice(ks)
                rks = [Replacement(m.start(), m.end(), entity, new, label) for m in word(entity).finditer(sentences[k])]
                return self._pair(sentences, j, k, [Replacement(start, end, entity, new, label)], rks, entity, "shared")
        return None

    def _alias(self, sentences, article, order):
        for j in order:
            for start, end, entity, label in self.entity_mutator._extract_entities(sentences[j]):
                parts = entity.split()
                if label != "PERSON" or len(parts) < 2:
                    continue
                surname = parts[-1]
                ks = [k for k in order if k != j and word(surname).search(sentences[k]) and entity not in sentences[k]]
                if not ks:
                    continue
                new = self.entity_mutator.replacement_for(entity, label, article)
                if not new or len(new.split()) < 2 or new.split()[-1] == surname:
                    continue
                k = self.rng.choice(ks)
                rks = [Replacement(m.start(), m.end(), surname, new.split()[-1], label)
                       for m in word(surname).finditer(sentences[k])]
                return self._pair(sentences, j, k, [Replacement(start, end, entity, new, label)], rks,
                                  f"{entity} / {surname}", "alias")
        return None

    def _demonym(self, sentences, article, order):
        for j in order:
            for country, demonym in DEMONYMS.items():
                mj = word(country).search(sentences[j])
                if not mj:
                    continue
                ks = [k for k in order if k != j and word(demonym).search(sentences[k]) and not word(country).search(sentences[k])]
                if not ks:
                    continue
                others = [c for c in DEMONYMS if c != country and not word(c).search(" ".join(sentences))]
                new = self.rng.choice(others)
                k = self.rng.choice(ks)
                rks = [Replacement(m.start(), m.end(), demonym, DEMONYMS[new], "NORP")
                       for m in word(demonym).finditer(sentences[k])]
                return self._pair(sentences, j, k, [Replacement(mj.start(), mj.end(), country, new, "GPE")], rks,
                                  f"{country} / {demonym}", "demonym")
        return None

    def find_and_inject_cascade(self, sentences: list[str], article: dict | None = None) -> CascadePair | None:
        """A direct edit j and a dependent edit k, or None when no strategy applies to the article."""
        if len(sentences) < 2:
            return None
        sentences = [s.strip() for s in sentences]
        order = list(range(len(sentences)))
        self.rng.shuffle(order)
        for strategy in self.rng.sample(STRATEGIES, len(STRATEGIES)):
            pair = getattr(self, f"_{strategy}")(sentences, article, order)
            if pair is not None:
                return pair
        return None
