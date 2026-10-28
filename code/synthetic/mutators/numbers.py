"""Number and date mutator for factual corruption.

Each kind of value moves by a random amount that keeps its written form:

    year       a shift of up to 30 years, small shifts more likely
    quantity   a factor between 0.4 and 2.5, away from 1, rounded to the
               original's significant digits, with its commas and decimals
    small int  another integer within 5 of it
    word       another number word within a few steps ("three" to "five")
    ordinal    a nearby ordinal ("third" to "second", "3rd" to "5th")
    month      another month
    day        another day of the month, next to a month name
"""

from dataclasses import dataclass
import math
import random
import re

from .entities import Replacement, apply

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
         "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred"]
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
            "eleventh", "twelfth"]
YEAR_SHIFTS = [1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]

YEAR = re.compile(r"\b(1[0-9]\d\d|20[0-2]\d)\b")
NUMBER = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w,])|(?<![\w.,])\d+(?:\.\d+)?(?![\w,])")
MONTH = "|".join(MONTHS)
DAY_FIRST = re.compile(rf"\b([1-9]|[12]\d|3[01]) (?:{MONTH})\b")
DAY_AFTER = re.compile(rf"\b(?:{MONTH}) ([1-9]|[12]\d|3[01])\b")
ORDINAL_DIGITS = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b")


@dataclass
class NumberMutation:
    original_text: str
    mutated_text: str
    stale_spans: list[tuple[int, int]]
    original_val: str
    mutated_val: str
    num_type: str


def suffix(n):
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def match_case(new, old):
    return new.capitalize() if old[:1].isupper() else new


class NumberMutator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def _year(self, val):
        shift = self.rng.choice(YEAR_SHIFTS)
        new = int(val) + self.rng.choice([-1, 1]) * shift
        return str(new if new <= 2025 else int(val) - shift)

    def _quantity(self, val):
        plain = val.replace(",", "")
        decimals = len(plain.split(".")[1]) if "." in plain else 0
        x = float(plain)
        if x == 0:
            return None
        if x < 20 and decimals == 0:
            options = [n for n in range(max(1, int(x) - 5), int(x) + 6) if n != int(x)]
            return str(self.rng.choice(options))
        factor = self.rng.choice([self.rng.uniform(0.4, 0.85), self.rng.uniform(1.15, 2.5)])
        significant = max(1, min(3, len(plain.split(".")[0].rstrip("0"))))
        new = x * factor
        if decimals == 0:
            scale = 10 ** max(0, int(math.log10(new)) + 1 - significant)
            new = max(scale, round(new / scale) * scale)
            text = f"{int(new):,}" if "," in val else str(int(new))
        else:
            text = f"{new:,.{decimals}f}" if "," in val else f"{new:.{decimals}f}"
        return text if text != val else None

    def _word(self, val):
        i = WORDS.index(val.lower())
        options = [w for w in WORDS[max(0, i - 3):i + 4] if w != val.lower()]
        return match_case(self.rng.choice(options), val)

    def _ordinal(self, val):
        i = ORDINALS.index(val.lower())
        options = [w for w in ORDINALS[max(0, i - 2):i + 3] if w != val.lower()]
        return match_case(self.rng.choice(options), val)

    def candidates(self, text: str) -> list[Replacement]:
        """One possible replacement per number, date part or number word of `text`."""
        out, taken = [], []

        def add(start, end, val, new, kind):
            if new and new != val and not any(start < b and end > a for a, b in taken):
                out.append(Replacement(start, end, val, new, kind))
                taken.append((start, end))

        for m in YEAR.finditer(text):
            add(m.start(), m.end(), m.group(), self._year(m.group()), "year")
        for m in [*DAY_FIRST.finditer(text), *DAY_AFTER.finditer(text)]:
            day = int(m.group(1))
            new = str(self.rng.choice([d for d in range(1, 29) if d != day]))
            add(m.start(1), m.end(1), m.group(1), new, "day")
        for m in ORDINAL_DIGITS.finditer(text):
            n = int(m.group(1))
            new = self.rng.choice([k for k in range(max(1, n - 3), n + 4) if k != n])
            add(m.start(), m.end(), m.group(), f"{new}{suffix(new)}", "ordinal")
        for m in NUMBER.finditer(text):
            add(m.start(), m.end(), m.group(), self._quantity(m.group()), "quantity")
        for m in re.finditer(rf"\b(?:{'|'.join(MONTHS)})\b", text):
            add(m.start(), m.end(), m.group(), self.rng.choice([x for x in MONTHS if x != m.group()]), "month")
        for m in re.finditer(rf"\b(?:{'|'.join(WORDS)})\b", text, re.IGNORECASE):
            add(m.start(), m.end(), m.group(), self._word(m.group()), "word")
        for m in re.finditer(rf"\b(?:{'|'.join(ORDINALS)})\b", text, re.IGNORECASE):
            add(m.start(), m.end(), m.group(), self._ordinal(m.group()), "ordinal")
        return out

    def mutate(self, text: str) -> NumberMutation | None:
        """Change one number, date part or number word of `text`, or None when it has none."""
        stripped = text.strip()
        options = self.candidates(stripped)
        if not options:
            return None
        r = self.rng.choice(options)
        mutated, spans = apply(stripped, [r])
        return NumberMutation(stripped, mutated, spans, r.original, r.replacement, r.kind)
