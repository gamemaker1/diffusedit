"""Evidence snippets in varied forms for the synthetic generator.

A snippet states a fact of the corrected article, retracts a sentence, or
confirms one. Every kind uses the same pool of titles and frames, so the form
of a snippet does not reveal which label it serves.

    restatement  the whole sentence after a lead-in ("Records held by the
                 city archive show that ...") or none
    excerpt      a window of words around the fact, cut at word boundaries,
                 often with ellipses
    field        "<words before the fact>: <the fact>", as in a fact sheet
    retraction   a notice that a sentence is false or withdrawn, quoting the
                 sentence or a part of it
    confirmation the same notice with a confirming verb, for KEEP sentences

Titles name the article's main entity, a document type, both, or a year.
"""

import random
import re

DOCUMENTS = [
    "annual report", "press release", "news brief", "archive record", "museum catalogue", "census table",
    "court filing", "interview transcript", "obituary", "encyclopedia entry", "travel guide", "field notes",
    "conference abstract", "government gazette", "letter to the editor", "parliamentary record",
    "company filing", "university bulletin", "exhibition notes", "almanac entry", "society newsletter",
    "survey report", "radio broadcast", "sports yearbook", "trade journal", "heritage register",
]
SOURCES = [
    "the national archive", "a regional newspaper", "the official register", "a later survey",
    "the organisation's own records", "a contemporary account", "the census office", "a biographer",
    "local historians", "the ministry", "a trade publication", "the yearbook", "court documents",
    "the museum's catalogue", "an interview with a former member", "the municipal records",
]
LEADS = [
    "", "", "", "According to {source}, ", "{Source} reports that ", "Records held by {source} show that ",
    "As listed by {source}, ", "It is recorded that ", "One account notes that ", "Per {source}, ",
    "{Source} states that ", "Contemporary sources agree that ", "A note in {source} says that ",
]
TITLES = [
    "{entity}", "{entity}: {document}", "{Document}", "{Document} on {entity}", "{entity} ({year})",
    "Notes on {entity}", "{entity} in brief", "{Document}, {year}", "From the {document}", "{entity} profile",
]
NOTICES = [
    ("A {document} {verb} the statement '{quote}'", "withdrew", "upheld"),
    ("The claim that '{quote}' was {verb} after review.", "retracted", "confirmed"),
    ("Editors have {verb} the line '{quote}'", "struck", "kept"),
    ("'{quote}' This passage is {verb} by {source}.", "disputed", "supported"),
    ("{Source} lists '{quote}' as {verb}.", "erroneous", "accurate"),
    ("Erratum notes: the sentence '{quote}' is {verb}.", "incorrect", "correct"),
    ("A check against {source} found '{quote}' to be {verb}.", "unfounded", "sound"),
    ("No record supports '{quote}', and {source} {verb} it.", "rejected", "accepted"),
]
LOWER = {"The", "A", "An", "It", "In", "On", "At", "He", "She", "They", "This", "These", "Its", "His", "Her",
         "After", "Before", "During", "Following", "As", "By", "From", "With", "There", "Since", "When", "While"}


def lowered(sentence):
    first = sentence.split(" ", 1)[0]
    return sentence[0].lower() + sentence[1:] if first in LOWER else sentence


def fill(template, rng, **kw):
    source, document = rng.choice(SOURCES), rng.choice(DOCUMENTS)
    values = {"source": source, "Source": source[0].upper() + source[1:], "document": document,
              "Document": document[0].upper() + document[1:], **kw}
    return template.format(**values)


class Writer:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def title(self, entity: str | None, year: str | None) -> str:
        options = [t for t in TITLES if ("{entity}" not in t or entity) and ("{year}" not in t or year)]
        return fill(self.rng.choice(options), self.rng, entity=entity, year=year)

    def window(self, sentence: str, focus: list[tuple[int, int]] | None, words=(5, 14)) -> str:
        """A run of whole words of `sentence` that covers `focus`, or a random run."""
        tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", sentence)]
        n = len(tokens)
        size = min(n, self.rng.randint(*words))
        if focus:
            first = min(i for i, (a, b) in enumerate(tokens) if b > focus[0][0])
            last = max(i for i, (a, b) in enumerate(tokens) if a < focus[-1][1])
            size = max(size, last - first + 1)
            start = self.rng.randint(max(0, last - size + 1), min(first, n - size))
        else:
            start = self.rng.randint(0, n - size)
        text = sentence[tokens[start][0]:tokens[start + size - 1][1]]
        if start > 0 and self.rng.random() < 0.6:
            text = "... " + text
        if start + size < n and self.rng.random() < 0.6:
            text = text.rstrip(".,;:") + " ..."
        return text

    def support(self, sentence: str, focus: list[tuple[int, int]] | None = None) -> str:
        """A snippet stating the fact of `sentence`, focused on the spans `focus` when given."""
        form = self.rng.choices(["restatement", "excerpt", "field"], weights=[0.5, 0.35, 0.15 if focus else 0])[0]
        if form == "restatement":
            lead = fill(self.rng.choice(LEADS), self.rng)
            return lead + (lowered(sentence) if lead else sentence)
        if form == "excerpt":
            return self.window(sentence, focus)
        a, b = focus[0][0], focus[-1][1]
        key = " ".join(sentence[:a].split()[-self.rng.randint(2, 4):]).strip(" ,;:(") or "Recorded"
        return f"{key[0].upper()}{key[1:]}: {sentence[a:b]}."

    def notice(self, sentence: str, retract: bool) -> str:
        """A retraction of `sentence`, or with retract=False a confirmation in the same frame."""
        template, no, yes = self.rng.choice(NOTICES)
        quote = sentence.rstrip(".") if self.rng.random() < 0.6 else self.window(sentence, None, (4, 9)).strip(". ")
        return fill(template, self.rng, quote=quote, verb=no if retract else yes)
