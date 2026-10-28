"""Cascading edit mutator for coupled factual dependencies (s_j -> s_k).

Sentence j receives a direct error that the evidence corrects. Sentence k
receives an error that follows from the one in j. The fix of k is a string
that the fix of j does not contain, so evidence that states only j's fact
does not state k's, and k can be repaired only by reasoning from j. Each
strategy changes text the seed article already holds, so no template
sentence is added. The strategies are tried in random order.

    century   j holds a year and k the century it falls in ("1848", "19th
              century"). j's year moves by one or two centuries and k's
              century follows.
    decade    j holds a year and k its decade ("1848", "the 1840s"). The
              year moves by one to three decades and k's decade follows.
    location  j names a city and k its country or the country's demonym
              ("Lyon", "France"). j names a city of another country and k
              that country.
    demonym   j names a country and k uses its demonym ("France", "French").
              j names another country and k its demonym.

In every strategy j must not hold k's string, and k must not hold j's, so
neither sentence gives the other's fix away.
"""

from dataclasses import dataclass
import random
import re

from .entities import EntityBank, Replacement, apply
from .numbers import YEAR, suffix

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
    "Wales": "Welsh", "United States": "American", "Netherlands": "Dutch", "Philippines": "Filipino",
    "South Africa": "South African", "South Korea": "South Korean", "New Zealand": "New Zealand",
    "Czech Republic": "Czech", "Slovakia": "Slovak", "Slovenia": "Slovenian", "Venezuela": "Venezuelan",
    "Uruguay": "Uruguayan", "Ecuador": "Ecuadorian", "Bolivia": "Bolivian", "Ethiopia": "Ethiopian",
}
CITIES = {
    "Argentina": ["Buenos Aires", "Córdoba", "Rosario", "Mendoza"], "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"],
    "Austria": ["Vienna", "Graz", "Linz", "Salzburg"], "Belgium": ["Brussels", "Antwerp", "Ghent", "Liège"],
    "Brazil": ["São Paulo", "Rio de Janeiro", "Salvador", "Recife"], "Canada": ["Toronto", "Montreal", "Vancouver", "Calgary", "Ottawa"],
    "Chile": ["Santiago", "Valparaíso", "Concepción"], "China": ["Beijing", "Shanghai", "Guangzhou", "Wuhan", "Chengdu"],
    "Colombia": ["Bogotá", "Medellín", "Cali"], "Denmark": ["Copenhagen", "Aarhus", "Odense"], "Egypt": ["Cairo", "Alexandria", "Giza"],
    "England": ["London", "Manchester", "Birmingham", "Liverpool", "Leeds", "Bristol"], "Finland": ["Helsinki", "Tampere", "Turku"],
    "France": ["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux", "Lille", "Nantes"],
    "Germany": ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt", "Stuttgart", "Dresden", "Leipzig"],
    "Greece": ["Athens", "Thessaloniki", "Patras"], "Hungary": ["Budapest", "Debrecen", "Szeged"],
    "India": ["Mumbai", "Delhi", "Kolkata", "Chennai", "Bangalore", "Hyderabad"], "Indonesia": ["Jakarta", "Surabaya", "Bandung"],
    "Iran": ["Tehran", "Isfahan", "Tabriz"], "Ireland": ["Dublin", "Cork", "Galway", "Limerick"], "Israel": ["Jerusalem", "Tel Aviv", "Haifa"],
    "Italy": ["Rome", "Milan", "Naples", "Turin", "Florence", "Venice", "Bologna", "Genoa"], "Japan": ["Tokyo", "Osaka", "Kyoto", "Yokohama", "Nagoya"],
    "Kenya": ["Nairobi", "Mombasa"], "Mexico": ["Mexico City", "Guadalajara", "Monterrey", "Puebla"], "Netherlands": ["Amsterdam", "Rotterdam", "Utrecht", "The Hague"],
    "Nigeria": ["Lagos", "Abuja", "Kano", "Ibadan"], "Norway": ["Oslo", "Bergen", "Trondheim"], "Pakistan": ["Karachi", "Lahore", "Islamabad"],
    "Peru": ["Lima", "Arequipa", "Cusco"], "Poland": ["Warsaw", "Kraków", "Gdańsk", "Wrocław", "Poznań"], "Portugal": ["Lisbon", "Porto", "Coimbra"],
    "Romania": ["Bucharest", "Cluj-Napoca", "Iași"], "Russia": ["Moscow", "Saint Petersburg", "Novosibirsk", "Kazan"],
    "Scotland": ["Edinburgh", "Glasgow", "Aberdeen", "Dundee"], "Spain": ["Madrid", "Barcelona", "Valencia", "Seville", "Bilbao"],
    "Sweden": ["Stockholm", "Gothenburg", "Malmö", "Uppsala"], "Switzerland": ["Zurich", "Geneva", "Basel", "Bern"],
    "Turkey": ["Istanbul", "Ankara", "Izmir"], "Ukraine": ["Kyiv", "Kharkiv", "Odesa", "Lviv"], "Wales": ["Cardiff", "Swansea", "Newport"],
    "United States": ["New York City", "Chicago", "Los Angeles", "Boston", "Philadelphia", "San Francisco", "Seattle"],
    "South Africa": ["Johannesburg", "Cape Town", "Durban"], "South Korea": ["Seoul", "Busan", "Incheon"],
    "Philippines": ["Manila", "Cebu City", "Davao City"], "Czech Republic": ["Prague", "Brno", "Ostrava"],
}
CENTURY_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                 "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
                 "eighteenth", "nineteenth", "twentieth", "twenty-first"]


def disjoint(replacements):
    """The replacements in text order, without any that overlaps an earlier one."""
    out = []
    for r in sorted(replacements, key=lambda r: (r.start, -r.end)):
        if not out or r.start >= out[-1].end:
            out.append(r)
    return out


def century(year):
    return (year - 1) // 100 + 1
STRATEGIES = ("century", "decade", "location", "demonym")


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

    def _pair(self, sentences, j, k, rj, rks, concept, strategy):
        mutated_j, spans_j = apply(sentences[j], disjoint(rj))
        mutated_k, spans_k = apply(sentences[k], disjoint(rks))
        return CascadePair(j, k, sentences[j], mutated_j, spans_j, sentences[k], mutated_k, spans_k, concept, strategy)

    def _century(self, sentences, article, order):
        text = " ".join(sentences)
        for j in order:
            for m in YEAR.finditer(sentences[j]):
                year = int(m.group())
                c = century(year)
                for render in (lambda n: f"{n}{suffix(n)}", lambda n: CENTURY_WORDS[n - 1]):
                    pattern = rf"(?<![\w-]){re.escape(render(c))}(?=[- ]century)"
                    if re.search(pattern, sentences[j], re.IGNORECASE):
                        continue
                    ks = [k for k in order if k != j and re.search(pattern, sentences[k], re.IGNORECASE)
                          and not any(century(int(y.group())) == c for y in YEAR.finditer(sentences[k]))]
                    shifts = [s for s in (-200, -100, 100, 200) if 1000 <= year + s <= 2025 and str(year + s) not in text]
                    if not ks or not shifts:
                        continue
                    new = year + self.rng.choice(shifts)
                    word_new = render(century(new))
                    k = self.rng.choice(ks)
                    rks = [Replacement(mk.start(), mk.end(), mk.group(),
                                       word_new[0].upper() + word_new[1:] if mk.group()[:1].isupper() else word_new, "DATE")
                           for mk in re.finditer(pattern, sentences[k], re.IGNORECASE)]
                    rj = [Replacement(m.start(), m.end(), m.group(), str(new), "DATE")]
                    return self._pair(sentences, j, k, rj, rks, f"{year} / century {c}", "century")
        return None

    def _decade(self, sentences, article, order):
        text = " ".join(sentences)
        for j in order:
            for m in YEAR.finditer(sentences[j]):
                year = int(m.group())
                decade = f"{year // 10 * 10}s"
                ks = [k for k in order if k != j and word(decade).search(sentences[k])
                      and not any(int(y.group()) // 10 == year // 10 for y in YEAR.finditer(sentences[k]))]
                if not ks or word(decade).search(sentences[j]):
                    continue
                shifts = [s for s in (-30, -20, -10, 10, 20, 30) if 1000 <= year + s <= 2025 and str(year + s) not in text
                          and f"{(year + s) // 10 * 10}s" not in text]
                if not shifts:
                    continue
                new = year + self.rng.choice(shifts)
                k = self.rng.choice(ks)
                rks = [Replacement(mk.start(), mk.end(), decade, f"{new // 10 * 10}s", "DATE")
                       for mk in word(decade).finditer(sentences[k])]
                rj = [Replacement(m.start(), m.end(), m.group(), str(new), "DATE")]
                return self._pair(sentences, j, k, rj, rks, f"{year} / {decade}", "decade")
        return None

    def _location(self, sentences, article, order):
        text = " ".join(sentences)
        for j in order:
            for country, cities in CITIES.items():
                for city in cities:
                    mj = word(city).search(sentences[j])
                    if not mj:
                        continue
                    names = [n for n in (country, DEMONYMS.get(country)) if n]
                    if any(word(n).search(sentences[j]) for n in names):
                        continue
                    ks = [k for k in order if k != j and not word(city).search(sentences[k])
                          and any(word(n).search(sentences[k]) for n in names)]
                    others = [c for c in CITIES if c != country and not word(c).search(text)
                              and not (DEMONYMS.get(c) and word(DEMONYMS[c]).search(text))]
                    if not ks or not others:
                        continue
                    new_country = self.rng.choice(others)
                    new_city = self.rng.choice(CITIES[new_country])
                    k = self.rng.choice(ks)
                    rks = [Replacement(mk.start(), mk.end(), country, new_country, "GPE")
                           for mk in word(country).finditer(sentences[k])]
                    if country in DEMONYMS and new_country in DEMONYMS:
                        rks += [Replacement(mk.start(), mk.end(), DEMONYMS[country], DEMONYMS[new_country], "NORP")
                                for mk in word(DEMONYMS[country]).finditer(sentences[k])]
                    rks = disjoint(rks)
                    if not rks:
                        continue
                    rj = [Replacement(mj.start(), mj.end(), city, new_city, "GPE")]
                    return self._pair(sentences, j, k, rj, rks, f"{city} / {country}", "location")
        return None

    def _demonym(self, sentences, article, order):
        text = " ".join(sentences)
        for j in order:
            for country, demonym in DEMONYMS.items():
                mj = word(country).search(sentences[j])
                if not mj or demonym == country or word(demonym).search(sentences[j]):
                    continue
                ks = [k for k in order if k != j and word(demonym).search(sentences[k]) and not word(country).search(sentences[k])]
                if not ks:
                    continue
                others = [c for c in DEMONYMS if c != country and DEMONYMS[c] != c and not word(c).search(text)
                          and not word(DEMONYMS[c]).search(text)]
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
