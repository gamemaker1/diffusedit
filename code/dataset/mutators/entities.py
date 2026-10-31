"""Named entity mutator for factual corruption.

Replacements come from the seed corpus itself. An EntityBank runs spaCy
en_core_web_sm once over every seed sentence and keeps the named entities of
each type. A replacement has the same type, the same word count and the same
use of digits as the entity it replaces. With probability SAME_ARTICLE it is
another entity of the same article, such as a second city the article names,
so the corrupted sentence stays plausible in context. Otherwise it comes from
another article. The fixed pools below serve only when spaCy is missing or a
type has no other candidate.
"""

from collections import defaultdict
from dataclasses import dataclass
import random
import re

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
except Exception:
    _NLP = None

# Entity types this mutator swaps. Dates, times and numbers belong to NumberMutator.
SWAPPABLE = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "WORK_OF_ART", "LANGUAGE", "PRODUCT"}
SAME_ARTICLE = 0.4

FALLBACK_POOLS = {
    "PERSON": ["Alexander Wright", "Marcus Chen", "Sofia Al-Mansoor", "Lucas Moreau", "Amina Diallo"],
    "ORG": ["Quantum Dynamics", "Vanguard Heritage Foundation", "Veritas Media Group"],
    "GPE": ["Munich", "Toronto", "Rotterdam", "Brisbane", "Nairobi", "Helsinki", "Marseille"],
    "LOC": ["the Baltic Sea", "Lake Baikal", "the Rhine Valley", "the Danube River"],
    "NORP": ["British", "Canadian", "Japanese", "Estonian", "Swedish", "Brazilian"],
}


@dataclass
class Replacement:
    """Replace text[start:end] (original) by replacement."""
    start: int
    end: int
    original: str
    replacement: str
    kind: str


@dataclass
class EntityMutation:
    original_text: str
    mutated_text: str
    stale_spans: list[tuple[int, int]]
    original_entity: str
    mutated_entity: str
    entity_type: str


def shape(text):
    return len(text.split()), any(c.isdigit() for c in text)


def apply(text, replacements):
    """The text with non-overlapping replacements applied, and their spans in the new text."""
    out, spans, pos, shift = [], [], 0, 0
    for r in sorted(replacements, key=lambda r: r.start):
        out.append(text[pos:r.start])
        start = r.start + shift
        out.append(r.replacement)
        spans.append((start, start + len(r.replacement)))
        shift += len(r.replacement) - (r.end - r.start)
        pos = r.end
    out.append(text[pos:])
    return "".join(out), spans


class EntityBank:
    """Named entities of the seed corpus, by type and by article, and the entities of each sentence."""

    def __init__(self, seeds, nlp=_NLP, batch_size=256):
        self.by_label = defaultdict(set)
        self.by_shape = {}
        self.sentence_entities = {}
        self.article_entities = []
        if nlp is None:
            for label, pool in FALLBACK_POOLS.items():
                self.by_label[label].update(pool)
            self.article_entities = [defaultdict(set) for _ in seeds]
            return
        sentences = sorted({s.strip() for seed in seeds for s in seed})
        for sentence, doc in zip(sentences, nlp.pipe(sentences, batch_size=batch_size)):
            ents = [(e.start_char, e.end_char, e.text, e.label_) for e in doc.ents]
            self.sentence_entities[sentence] = ents
            for _, _, text, label in ents:
                if label in SWAPPABLE:
                    self.by_label[label].add(text)
        for label in self.by_label:
            self.by_label[label] = set(self.by_label[label])
        self.by_shape = {label: defaultdict(list) for label in self.by_label}
        for label, texts in self.by_label.items():
            for text in sorted(texts):
                self.by_shape[label][shape(text)].append(text)
        for seed in seeds:
            local = defaultdict(set)
            for s in seed:
                for _, _, text, label in self.sentence_entities.get(s.strip(), []):
                    local[label].add(text)
            self.article_entities.append(local)

    def entities(self, sentence):
        """(start, end, text, label) of each entity of a seed sentence, or a fresh spaCy pass."""
        sentence = sentence.strip()
        if sentence not in self.sentence_entities and _NLP is not None:
            doc = _NLP(sentence)
            self.sentence_entities[sentence] = [(e.start_char, e.end_char, e.text, e.label_) for e in doc.ents]
        return self.sentence_entities.get(sentence, [])


class EntityMutator:
    def __init__(self, seed: int | None = None, bank: EntityBank | None = None):
        self.rng = random.Random(seed)
        self.nlp = _NLP
        self.bank = bank

    def _extract_entities(self, text: str) -> list[tuple[int, int, str, str]]:
        """(start, end, text, label) of each swappable entity. Uses spaCy or a capitalized-word fallback."""
        if self.bank is not None:
            ents = self.bank.entities(text)
        elif self.nlp is not None:
            ents = [(e.start_char, e.end_char, e.text, e.label_) for e in self.nlp(text).ents]
        else:
            ents = []
            for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text):
                if match.start() == 0 and len(match.group().split()) == 1:
                    continue
                ents.append((match.start(), match.end(), match.group(), "PERSON"))
        return [e for e in ents if e[3] in SWAPPABLE]

    def replacement_for(self, entity: str, label: str, article: dict | None = None, avoid: set | None = None):
        """An entity of the same type and shape, from the same article with probability SAME_ARTICLE."""
        avoid = {a.lower() for a in (avoid or set())} | {entity.lower()}
        key = shape(entity)
        if article and self.rng.random() < SAME_ARTICLE:
            local = sorted(t for t in article.get(label, ()) if t.lower() not in avoid and shape(t) == key)
            if local:
                return self.rng.choice(local)
        pool = []
        if self.bank is not None and label in self.bank.by_shape:
            pool = self.bank.by_shape[label].get(key, [])
            if len(pool) < 2:
                pool = sorted(self.bank.by_label[label])
        if len(pool) < 2:
            pool = FALLBACK_POOLS.get(label, FALLBACK_POOLS["PERSON"])
        for _ in range(20):
            choice = self.rng.choice(pool)
            if choice.lower() not in avoid:
                return choice
        return None

    def candidates(self, text: str, article: dict | None = None) -> list[Replacement]:
        """One possible replacement per swappable entity of `text`."""
        out = []
        for start, end, entity, label in self._extract_entities(text):
            new = self.replacement_for(entity, label, article)
            if new and new != entity:
                out.append(Replacement(start, end, entity, new, label))
        return out

    def mutate(self, text: str, article: dict | None = None) -> EntityMutation | None:
        """Replace one entity of `text`, or None when it has no swappable entity."""
        stripped = text.strip()
        options = self.candidates(stripped, article)
        if not options:
            return None
        r = self.rng.choice(options)
        mutated, spans = apply(stripped, [r])
        return EntityMutation(stripped, mutated, spans, r.original, r.replacement, r.kind)
