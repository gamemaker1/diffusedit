"""Mention extractor M(text) — Section 7.3.

Returns a set of lowercased mention strings from a piece of text:
  - Named entities of 13 types (spaCy en_core_web_sm)
  - Every digit string (numbers, years, statistics)

This is the same extractor used in infilling.entities and datasets/collate.py.
Centralised here so cascade detection, mining, and evaluation all share one
implementation.

Usage:
    from cascading.mentions import Extractor
    M = Extractor()
    mentions = M("Three people died in the fire.")  # {"three", "3"}  (if spaCy tags 'Three')
"""

import re

# Entity label types from the paper (Section 7.3)
MENTION_LABELS = {
    "CARDINAL", "DATE", "FAC", "GPE", "LOC", "MONEY",
    "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "QUANTITY", "TIME",
}
MENTION_STRIP = " .,;:'\"()!?\t\n"
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


class Extractor:
    """Thread-unsafe but cacheing mention extractor M(·) from Section 7.3.

    Lazily loads spaCy on first call so importing this module is always free.
    """

    def __init__(self, model: str = "en_core_web_sm", cache: bool = True):
        self._model_name = model
        self._nlp = None
        self._cache: dict[str, frozenset[str]] = {} if cache else None

    def _load(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self._model_name)
            except OSError:
                raise OSError(
                    f"spaCy model '{self._model_name}' not found. "
                    "Install it with: python -m spacy download en_core_web_sm"
                )

    def __call__(self, text: str) -> frozenset[str]:
        """Return the mention set M(text) as a frozenset of lowercase strings."""
        if self._cache is not None and text in self._cache:
            return self._cache[text]

        self._load()
        doc = self._nlp(text)
        out: set[str] = {
            re.sub(r"\s+", " ", e.text.lower()).strip(MENTION_STRIP)
            for e in doc.ents
            if e.label_ in MENTION_LABELS
        }
        # Always add raw digit strings regardless of NER tagging
        out |= set(NUMBER_RE.findall(text))
        out.discard("")
        result = frozenset(out)

        if self._cache is not None:
            self._cache[text] = result
        return result

    def from_evidence(self, evidence: list[dict]) -> frozenset[str]:
        """Union of mentions across all evidence snippets."""
        merged: set[str] = set()
        for snippet in evidence:
            merged |= self(f"{snippet.get('title', '')}: {snippet.get('text', '')}")
        return frozenset(merged)

    def changed_mentions(self, source: str, target: str) -> frozenset[str]:
        """D_j: mentions that appear in exactly one of {source, target} (Section 8.5)."""
        m_src = self(source)
        m_tgt = self(target)
        return (m_src | m_tgt) - (m_src & m_tgt)


# Module-level singleton for convenience
_default_extractor: Extractor | None = None


def get_extractor() -> Extractor:
    """Return (and lazily create) the module-level singleton Extractor."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = Extractor()
    return _default_extractor
