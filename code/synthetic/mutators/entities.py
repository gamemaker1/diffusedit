"""Named Entity mutator for factual corruption."""

from dataclasses import dataclass
import random
import re

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

# Entity substitution pools
ENTITY_POOLS = {
    "PERSON": [
        "Alexander Wright", "Elena Rostova", "Marcus Chen", "David Hughes",
        "Sofia Al-Mansoor", "Lucas Moreau", "Amina Diallo", "Klaus Becker",
        "Vikram Patel", "Claire Tremblay", "Hiroshi Tanaka", "Mateo Silva",
    ],
    "ORG": [
        "Apex Global Logistics", "Quantum Dynamics", "Vanguard Heritage Foundation",
        "Astraeus Aerospace", "Boreal Financial Group", "Kyoto Institute of Technology",
        "Veritas Media Group", "Solaria Energy Corp", "Aegis Health Systems",
    ],
    "GPE": [
        "Munich", "Toronto", "Kyoto", "Buenos Aires", "Rotterdam", "Brisbane",
        "Stockholm", "Valparaíso", "Nairobi", "Helsinki", "Marseille", "Osaka",
        "Germany", "Canada", "Japan", "Argentina", "Netherlands", "Australia",
    ],
    "LOC": [
        "the Baltic Sea", "the Appalachian Mountains", "Lake Baikal", "the Rhine Valley",
        "the Andes range", "the Sahara basin", "the Danube River",
    ],
    "EVENT": [
        "the 1998 Global Summit", "the Great Northern Expedition", "the Lisbon Festival",
        "the Pacific Games", "the International Congress of 1974",
    ],
    "NORP": [
        "British", "Canadian", "Japanese", "Estonian", "Argentine", "Swedish",
        "Brazilian", "Portuguese", "Danish", "Swiss", "Korean",
    ],
}


@dataclass
class EntityMutation:
    original_text: str
    mutated_text: str
    stale_spans: list[tuple[int, int]]
    original_entity: str
    mutated_entity: str
    entity_type: str


class EntityMutator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.nlp = _NLP

    def _extract_entities(self, text: str) -> list[tuple[int, int, str, str]]:
        """Extract (start, end, entity_text, label). Uses spaCy or regex fallback."""
        if self.nlp is not None:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                if ent.label_ in ENTITY_POOLS:
                    entities.append((ent.start_char, ent.end_char, ent.text, ent.label_))
            if entities:
                return entities

        # Heuristic / regex fallback for capitalized names and known patterns
        entities = []
        for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text):
            # Exclude start of sentence if single common word
            start, end = match.span()
            val = match.group()
            if start == 0 and len(val.split()) == 1 and val in ("The", "During", "In", "After", "Before", "It", "He", "She"):
                continue
            entities.append((start, end, val, "PERSON"))
        return entities

    def mutate(self, text: str) -> EntityMutation | None:
        """Perturb an entity in `text` to create an outdated/erroneous sentence.

        Returns an EntityMutation containing the mutated sentence, the character
        range of the mutated entity (stale_span), and the original entity string.
        """
        stripped = text.strip()
        entities = self._extract_entities(stripped)
        if not entities:
            return None

        # Choose one entity to mutate
        start, end, original_ent, label = self.rng.choice(entities)
        candidates = [c for c in ENTITY_POOLS.get(label, ENTITY_POOLS["PERSON"]) if c.lower() != original_ent.lower()]
        if not candidates:
            return None

        replacement = self.rng.choice(candidates)
        mutated_text = stripped[:start] + replacement + stripped[end:]
        stale_span = (start, start + len(replacement))

        return EntityMutation(
            original_text=stripped,
            mutated_text=mutated_text,
            stale_spans=[stale_span],
            original_entity=original_ent,
            mutated_entity=replacement,
            entity_type=label,
        )
