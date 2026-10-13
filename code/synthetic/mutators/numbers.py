"""Number and date mutator for factual corruption."""

from dataclasses import dataclass
import random
import re

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

WORD_NUMBERS = {
    "zero": "one", "one": "three", "two": "four", "three": "eight",
    "four": "six", "five": "nine", "six": "two", "seven": "five",
    "eight": "ten", "nine": "four", "ten": "twelve", "eleven": "seven",
    "twelve": "fifteen", "twenty": "fifty", "thirty": "sixty",
    "fifty": "eighty", "hundred": "thousand", "thousand": "million",
}


@dataclass
class NumberMutation:
    original_text: str
    mutated_text: str
    stale_spans: list[tuple[int, int]]
    original_val: str
    mutated_val: str
    num_type: str


class NumberMutator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def _find_years(self, text: str) -> list[tuple[int, int, str]]:
        """Find 4-digit years like 1850..2025."""
        matches = []
        for m in re.finditer(r"\b(1[789]\d\d|20[0-2]\d)\b", text):
            matches.append((m.start(), m.end(), m.group()))
        return matches

    def _find_quantities(self, text: str) -> list[tuple[int, int, str]]:
        """Find numbers, percentages, or money."""
        matches = []
        for m in re.finditer(r"\b(?:\$|£|€)?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:percent|%|million|billion|thousand))?\b", text):
            # Skip if already a 4-digit year
            val = m.group()
            if re.fullmatch(r"1[789]\d\d|20[0-2]\d", val):
                continue
            matches.append((m.start(), m.end(), val))
        return matches

    def _find_month_dates(self, text: str) -> list[tuple[int, int, str]]:
        """Find month names."""
        matches = []
        for month in MONTHS:
            for m in re.finditer(rf"\b{month}\b", text):
                matches.append((m.start(), m.end(), m.group()))
        return matches

    def _find_word_numbers(self, text: str) -> list[tuple[int, int, str]]:
        matches = []
        for word in WORD_NUMBERS:
            for m in re.finditer(rf"\b{word}\b", text, re.IGNORECASE):
                matches.append((m.start(), m.end(), m.group()))
        return matches

    def mutate(self, text: str) -> NumberMutation | None:
        """Perturb a number, year, or date in `text`."""
        stripped = text.strip()

        # Gather candidates
        years = self._find_years(stripped)
        quantities = self._find_quantities(stripped)
        months = self._find_month_dates(stripped)
        word_nums = self._find_word_numbers(stripped)

        types_available = []
        if years:
            types_available.append("year")
        if quantities:
            types_available.append("quantity")
        if months:
            types_available.append("month")
        if word_nums:
            types_available.append("word_num")

        if not types_available:
            return None

        chosen_type = self.rng.choice(types_available)

        if chosen_type == "year":
            start, end, val = self.rng.choice(years)
            year_int = int(val)
            delta = self.rng.choice([-5, -3, -1, 1, 3, 5, 10, 12])
            new_val = str(year_int + delta)
        elif chosen_type == "quantity":
            start, end, val = self.rng.choice(quantities)
            # Simple numeric scaling if pure digits
            num_match = re.search(r"\d+", val)
            if num_match:
                n = int(num_match.group())
                scale = self.rng.choice([2, 3, 5, 10])
                replacement_n = str(n * scale if n < 50 else max(1, n // scale))
                new_val = val[:num_match.start()] + replacement_n + val[num_match.end():]
            else:
                new_val = "99"
        elif chosen_type == "month":
            start, end, val = self.rng.choice(months)
            other_months = [m for m in MONTHS if m.lower() != val.lower()]
            new_val = self.rng.choice(other_months)
        else:  # word_num
            start, end, val = self.rng.choice(word_nums)
            new_val = WORD_NUMBERS.get(val.lower(), "seven")
            if val.istitle():
                new_val = new_val.capitalize()

        mutated_text = stripped[:start] + new_val + stripped[end:]
        stale_span = (start, start + len(new_val))

        return NumberMutation(
            original_text=stripped,
            mutated_text=mutated_text,
            stale_spans=[stale_span],
            original_val=val,
            mutated_val=new_val,
            num_type=chosen_type,
        )
