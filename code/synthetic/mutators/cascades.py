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
    age       j gives a birth year and k an age at a year that k states or
              at the death year that follows the birth year in j ("born
              1939", "at the age of 60 in 2000"). The birth year moves by up
              to 12 years and the age follows.
    duration  j gives a year and k the number of years from it to a year
              that k states, or an anniversary ("hired in 1972", "after 30
              years ... retired in 2002"). The year moves and the count
              follows. No other year of the article lies at the same
              distance.
    gender    j says what the subject is with a gendered noun and every
              other sentence about them uses pronouns ("an American
              actress", "She ..."). The noun takes the other gender, and so
              do the pronouns of every such k, so one j has up to three
              cascades. The subject's first name becomes a name of the other
              gender wherever it occurs, which must be in j or a cascade, so
              the name does not reveal the stale pronouns. The article holds
              no other gendered noun, no honorific and no pronoun of the
              other gender.
    field     j names the subject's occupation and up to three other
              sentences name its field ("physicist", "physics"). The
              occupation becomes another of its family (groups of similar
              sports, sciences, instruments, arts) and the field words
              follow.

In every strategy j must not hold k's string, and k must not hold j's, so
neither sentence gives the other's fix away. A pair is also rejected when the
correct string of k occurs in any other sentence of the article, since the
article would then state k's fix itself.
"""

from dataclasses import dataclass, field
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

ONES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
NUMBER = (r"(?P<n>\d{1,3}|(?:%s)(?:-(?:%s))?|%s)"
          % ("|".join(TENS), "|".join(ONES[:9]), "|".join(sorted(ONES, key=len, reverse=True))))
AGE = [re.compile(p, re.IGNORECASE) for p in (
    rf"\b(?:aged|at the age of|at age|age) {NUMBER}\b", rf"\b{NUMBER}(?:-year-old| years old)\b")]
# In "10th anniversary" the group n is the ordinal with its suffix.
DURATION = [re.compile(p, re.IGNORECASE) for p in (
    rf"\b{NUMBER} years? (?:later|earlier|after|before|afterwards|prior)\b",
    rf"\b(?:for|after|over|within|spanning|lasting|of) (?:almost |nearly |about |some |over )?{NUMBER} years\b",
    rf"\b{NUMBER}-year\b(?!-old)",
    r"\b(?P<n>\d{1,3}(?:st|nd|rd|th)) anniversary\b")]

# (male, female) role nouns that the gender kind flips. Nouns that are also
# common in names, titles or other senses ("man", "lord", "host", "count")
# are left out. The -man compounds are not flipped, because their -woman forms
# are rare in the seed text, so a flipped "businesswoman" would almost always
# be stale. They still count as gendered nouns of an article.
GENDERED = [("actor", "actress"), ("king", "queen"), ("prince", "princess"), ("duke", "duchess"),
            ("emperor", "empress"), ("baron", "baroness"), ("son", "daughter"), ("brother", "sister"),
            ("father", "mother"), ("husband", "wife"), ("nephew", "niece"), ("grandson", "granddaughter"),
            ("uncle", "aunt"), ("godfather", "godmother"), ("schoolboy", "schoolgirl"), ("monk", "nun"),
            ("waiter", "waitress")]
COMPOUNDS = [(f"{w}man", f"{w}woman") for w in ("business", "sports", "chair", "congress", "spokes", "council",
                                                 "police", "horse", "crafts", "states", "front")]
OPPOSITE = {**dict(GENDERED), **{f: m for m, f in GENDERED}}
FEMALE = {f for _, f in GENDERED}
GENDERED_WORDS = set(OPPOSITE) | {w for pair in COMPOUNDS for w in pair}
# First names that the seed biographies use for one gender only. The gender
# kind gives the subject a name of the gender it swaps to, so the name does
# not show which pronouns are stale.
FIRST_NAMES = {
    "M": ["Robert", "James", "David", "Michael", "Thomas", "Richard", "Charles", "Paul", "Joseph", "Mark", "Edward",
          "Daniel", "Andrew", "Brian", "Stephen", "Alexander", "Christopher", "Walter", "Henry", "Matthew", "Arthur",
          "Kevin", "Harry", "Steve", "Ronald", "Albert", "Alan", "Eric", "Frank", "Scott", "Alfred", "Donald", "Philip",
          "Samuel", "Jason", "Jeffrey", "Jonathan", "Patrick", "Tom", "Jack", "Juan", "Anthony", "Benjamin", "Gary",
          "Hans", "Fred", "Roger", "Carl", "Howard", "Bruce", "Mohammad", "Neil", "Adam", "Kenneth", "Stuart", "Carlos",
          "Ralph", "Louis", "Lawrence", "Hugh", "Ian", "Victor", "Herbert", "Frederick", "Mike", "Johann", "Tony"],
    "F": ["Mary", "Lisa", "Anna", "Anne", "Maria", "Jessica", "Barbara", "Susan", "Jennifer", "Sarah", "Kate",
          "Elizabeth", "Helen", "Patricia", "Nancy", "Margaret", "Ann", "Catherine", "Julia", "Jane", "Suzanne",
          "Dorothy", "Cynthia", "Caroline", "Stephanie", "Laura", "Carol", "Sharon", "Judith", "Louise", "Linda",
          "Nicole", "Amanda", "Christine", "Sophie", "Rebecca", "Hannah", "Diane", "Alice", "Amy", "Sara", "Jill",
          "Claire", "Pamela", "Edna", "Kathryn", "Rachel", "Natalie", "Angela", "Cheryl", "Katherine", "Ruth", "Amelia",
          "Frances", "Allison", "Alexandra", "Cathy", "Olga", "Irene", "Vanessa", "Michelle", "Lucy", "Kimberly",
          "Megan", "Elena", "Holly", "Sally"],
}
# The first name of a sentence that opens with a person's name: "Laura Johnson is ...",
# 'Edward "Gunboat" Smith was ...', "Darío Grandinetti is ...". NOT_NAMES are capitalized
# words that open such sentences without being a first name.
SUBJECT = re.compile(r"""^([A-ZÀ-Þ][^\W\d_]{2,})(?=\s+(?:"[^"]+"\s+)?(?:[A-ZÀ-Þ][\w.'-]*\s+){0,3}[A-ZÀ-Þ][^\W\d_]+)""")
NOT_NAMES = {"The", "This", "These", "That", "His", "Her", "Its", "Saint", "King", "Queen", "Prince", "Princess", "Lord",
             "Lady", "Pope", "Emperor", "Empress", "General", "Captain", "Major", "Colonel", "Admiral", "Doctor",
             "Professor", "Reverend", "Father", "Mother", "Sister", "Brother", "Bishop", "Cardinal", "President",
             "Senator", "Governor", "Judge", "Chief", "Duke", "Duchess", "Count", "Countess", "Baron", "Baroness",
             "Sheikh", "Sultan", "Rabbi", "Imam", "Saint", "Blessed", "Little", "Big", "Young", "Old", "Uncle", "Aunt"}
PRONOUNS = {"M": re.compile(r"\b(?:he|him|his|himself)\b", re.IGNORECASE),
            "F": re.compile(r"\b(?:she|her|hers|herself)\b", re.IGNORECASE)}
HONORIFIC = re.compile(r"\b(?:Sir|Dame|Mr|Mrs|Ms|Miss|Madame|Lady|Lord)\b")
# Words after which "her" is an object ("told her to"), not a possessive.
OBJECT_NEXT = {"to", "and", "or", "but", "as", "in", "on", "at", "for", "from", "with", "by", "into", "that", "the",
               "a", "an", "of", "after", "before", "during", "until", "when", "while", "because", "up", "out",
               "off", "back", "down", "again", "later", "there", "here", "away", "over", "about", "through"}

# Occupations and the words of their field, in families of interchangeable
# occupations. Sports are grouped so that the rest of an article about a
# striker or a race still fits the new sport. Occupations whose field word is
# common in other senses ("historian" and "history", "playwright" and
# "plays") are left out.
FIELDS = {
    "team sport": {"footballer": ["football"], "rugby union player": ["rugby union"],
                   "rugby league player": ["rugby league"], "handball player": ["handball"],
                   "ice hockey player": ["ice hockey"], "basketball player": ["basketball"],
                   "volleyball player": ["volleyball"], "baseball player": ["baseball"], "cricketer": ["cricket"]},
    "combat sport": {"boxer": ["boxing"], "wrestler": ["wrestling"], "judoka": ["judo"], "fencer": ["fencing"]},
    "racing": {"cyclist": ["cycling"], "swimmer": ["swimming"], "rower": ["rowing"], "skier": ["skiing"],
               "speed skater": ["speed skating"], "racing driver": ["motor racing"], "jockey": ["horse racing"]},
    "game": {"golfer": ["golf"], "archer": ["archery"], "snooker player": ["snooker"], "chess player": ["chess"],
             "tennis player": ["tennis"], "badminton player": ["badminton"]},
    "artistic sport": {"gymnast": ["gymnastics"], "figure skater": ["figure skating"], "weightlifter": ["weightlifting"]},
    "science": {"physicist": ["physics"], "chemist": ["chemistry"], "biologist": ["biology"],
                "mathematician": ["mathematics"], "economist": ["economics"], "philosopher": ["philosophy"],
                "linguist": ["linguistics"], "astronomer": ["astronomy"], "geologist": ["geology"],
                "sociologist": ["sociology"], "psychologist": ["psychology"], "anthropologist": ["anthropology"],
                "archaeologist": ["archaeology"], "botanist": ["botany"], "zoologist": ["zoology"],
                "geographer": ["geography"], "theologian": ["theology"], "neuroscientist": ["neuroscience"],
                "ecologist": ["ecology"], "geneticist": ["genetics"], "physiologist": ["physiology"],
                "astrophysicist": ["astrophysics"], "entomologist": ["entomology"], "ornithologist": ["ornithology"],
                "musicologist": ["musicology"], "criminologist": ["criminology"]},
    "music": {"pianist": ["piano"], "violinist": ["violin"], "guitarist": ["guitar"], "drummer": ["drums"],
              "cellist": ["cello"], "bassist": ["bass"], "saxophonist": ["saxophone"], "trumpeter": ["trumpet"],
              "flautist": ["flute"], "harpist": ["harp"], "clarinetist": ["clarinet"], "violist": ["viola"],
              "trombonist": ["trombone"], "oboist": ["oboe"], "percussionist": ["percussion"],
              "keyboardist": ["keyboards"], "banjoist": ["banjo"]},
    "art": {"painter": ["painting", "paintings"], "sculptor": ["sculpture", "sculptures"],
            "photographer": ["photography", "photographs"], "poet": ["poetry", "poems"],
            "architect": ["architecture"], "novelist": ["novels"], "illustrator": ["illustrations"],
            "ceramicist": ["ceramics"], "cartoonist": ["cartoons"], "choreographer": ["choreography"],
            "screenwriter": ["screenplays"]},
}


def role(nouns):
    """A sentence that says what its subject is or was: "is an American actress", "was the son of"."""
    alternatives = "|".join(sorted(map(re.escape, nouns), key=len, reverse=True))
    return re.compile(rf"\b(?:is|was|became)\s+(?:a|an|the)\s+(?:[\w'-]+,?\s+){{0,6}}?(?P<noun>{alternatives})\b(?!-)")


ROLE_GENDER = role(OPPOSITE)
ROLE_FIELD = {family: role(occupations) for family, occupations in FIELDS.items()}


def find(patterns, text):
    """The matches of any of `patterns` in `text`, in order and without overlaps."""
    out = []
    for m in sorted((m for p in patterns for m in p.finditer(text)), key=lambda m: (m.start(), -m.end())):
        if not out or m.start() >= out[-1].end():
            out.append(m)
    return out


def number_value(text):
    text = re.sub(r"(?<=\d)(?:st|nd|rd|th)$", "", text.lower())
    if text.isdigit():
        return int(text)
    if text in ONES:
        return ONES.index(text) + 1
    tens, _, ones = text.partition("-")
    return (TENS.index(tens) + 2) * 10 + (ONES.index(ones) + 1 if ones else 0)


def number_text(n, like):
    """n written as `like` is written: digits, or words with the same capital."""
    if like.isdigit() or n < 1 or n > 99:
        return str(n)
    text = ONES[n - 1] if n < 20 else TENS[n // 10 - 2] + (f"-{ONES[n % 10 - 1]}" if n % 10 else "")
    return text.capitalize() if like[:1].isupper() else text


def same_case(new, old):
    return new[:1].upper() + new[1:] if old[:1].isupper() else new


def swap_pronoun(match, sentence):
    """The pronoun of the other gender for `match`, reading "her" as an object or a possessive by the next word."""
    old = match.group()
    rest = sentence[match.end():]
    following = re.match(r"\s*([A-Za-z']+)", rest)
    last = not following or re.match(r"\s*[.,;:!?)]", rest)
    low = old.lower()
    if low == "her":
        new = "him" if last or following.group(1).lower() in OBJECT_NEXT else "his"
    elif low == "his":
        new = "hers" if last else "her"
    else:
        new = {"he": "she", "him": "her", "himself": "herself", "she": "he", "hers": "his", "herself": "himself"}[low]
    return same_case(new, old)


def disjoint(replacements):
    """The replacements in text order, without any that overlaps an earlier one."""
    out = []
    for r in sorted(replacements, key=lambda r: (r.start, -r.end)):
        if not out or r.start >= out[-1].end:
            out.append(r)
    return out


def fix_elsewhere(sentences, pair):
    """True when a correct string of the cascade fixes occurs in a sentence that is not a cascade."""
    others = " ".join(s for i, s in enumerate(sentences) if i not in pair.cascade_edits)
    return any(re.search(rf"(?<!\w){re.escape(f)}(?!\w)", others, re.IGNORECASE) for f in pair.cascade_fixes)


def century(year):
    return (year - 1) // 100 + 1


STRATEGIES = ("century", "decade", "location", "demonym", "age", "duration", "gender", "field")


@dataclass
class CascadePair:
    direct_sentence_idx: int
    direct_original: str
    direct_mutated: str
    direct_stale_spans: list[tuple[int, int]]
    # {k: (mutated sentence, stale spans)} for each dependent sentence k of j.
    cascade_edits: dict[int, tuple[str, list[tuple[int, int]]]]
    shared_concept: str
    strategy: str
    # The correct strings of the cascade fixes. No evidence may hold them.
    cascade_fixes: list[str] = field(default_factory=list)


def word(text):
    return re.compile(rf"(?<!\w){re.escape(text)}(?!\w)")


class CascadeMutator:
    def __init__(self, seed: int | None = None, bank: EntityBank | None = None, strategies=STRATEGIES):
        self.rng = random.Random(seed)
        self.bank = bank
        self.strategies = tuple(strategies)

    def _pair(self, sentences, j, rj, edits, concept, strategy):
        """The pair of direct edit j (replacements rj) and the cascades {k: replacements of k}."""
        mutated_j, spans_j = apply(sentences[j], disjoint(rj))
        cascade_edits = {k: apply(sentences[k], disjoint(rks)) for k, rks in sorted(edits.items())}
        # Strings that j's fix also corrects, such as a renamed subject, are not cascade fixes.
        fixes = sorted({r.original for rks in edits.values() for r in disjoint(rks)} - {r.original for r in rj})
        return CascadePair(j, sentences[j], mutated_j, spans_j, cascade_edits, concept, strategy, fixes)

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
                    return self._pair(sentences, j, rj, {k: rks}, f"{year} / century {c}", "century")
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
                return self._pair(sentences, j, rj, {k: rks}, f"{year} / {decade}", "decade")
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
                    return self._pair(sentences, j, rj, {k: rks}, f"{city} / {country}", "location")
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
                return self._pair(sentences, j, [Replacement(mj.start(), mj.end(), country, new, "GPE")], {k: rks},
                                  f"{country} / {demonym}", "demonym")
        return None

    def _age(self, sentences, article, order):
        """j gives a birth year and k an age at a year that k states, or at the death year that follows in j."""
        text = " ".join(sentences)
        for j in order:
            m = YEAR.search(sentences[j])
            if not m or not re.search(r"\bborn\b|\(", sentences[j][:m.start()]) or len(word(m.group()).findall(text)) != 1:
                continue
            birth = int(m.group())
            death = re.match(r"\s*[–-]\s*(?:[A-Za-z]+ \d{1,2}, |\d{1,2} [A-Za-z]+ )?(\d{4})\b", sentences[j][m.end():])
            for k in order:
                if k == j:
                    continue
                anchors = [int(y.group()) for y in YEAR.finditer(sentences[k])] + ([int(death.group(1))] if death else [])
                ages = [a for a in find(AGE, sentences[k])
                        if any(y - birth in (number_value(a.group("n")), number_value(a.group("n")) + 1) for y in anchors)]
                if len(ages) != 1:
                    continue
                a = ages[0]
                n = number_value(a.group("n"))
                shifts = [s for s in range(-12, 13) if s and 5 <= n - s <= 105 and not word(str(birth + s)).search(text)
                          and not word(number_text(n - s, a.group("n"))).search(sentences[k])]
                if not shifts:
                    continue
                s = self.rng.choice(shifts)
                rj = [Replacement(m.start(), m.end(), m.group(), str(birth + s), "DATE")]
                rk = [Replacement(a.start("n"), a.end("n"), a.group("n"), number_text(n - s, a.group("n")), "CARDINAL")]
                return self._pair(sentences, j, rj, {k: rk}, f"{birth} / age {n}", "age")
        return None

    def _duration(self, sentences, article, order):
        """j gives a year, and k counts the years, or the anniversary, from it to a year that k states."""
        text = " ".join(sentences)
        for j in order:
            for m in YEAR.finditer(sentences[j]):
                start = int(m.group())
                if len(word(m.group()).findall(text)) != 1:
                    continue
                for k in order:
                    if k == j:
                        continue
                    ends = [int(y.group()) for y in YEAR.finditer(sentences[k])]
                    for d in find(DURATION, sentences[k]):
                        old = d.group("n")
                        ordinal = not old.isalpha() and not old.isdigit()
                        n = number_value(old)
                        matches = [e for e in ends if (e - start == n if ordinal else abs(e - start) == n)]
                        # Another year of the article at the same distance would make the link ambiguous.
                        rivals = [y for y in YEAR.finditer(text) if int(y.group()) != start
                                  and any(abs(e - int(y.group())) == n for e in ends)]
                        if len(matches) != 1 or rivals:
                            continue
                        end = matches[0]
                        sign = 1 if end > start else -1
                        shifts = [s for s in range(-12, 13) if s and sign * (end - start - s) >= 1
                                  and abs(end - start - s) != n and not word(str(start + s)).search(text)]
                        if not shifts:
                            continue
                        s = self.rng.choice(shifts)
                        new = abs(end - start - s)
                        written = f"{new}{suffix(new)}" if ordinal else number_text(new, old)
                        rk = [Replacement(d.start("n"), d.end("n"), old, written, "CARDINAL")]
                        rj = [Replacement(m.start(), m.end(), m.group(), str(start + s), "DATE")]
                        return self._pair(sentences, j, rj, {k: rk}, f"{start} / {old}", "duration")
        return None

    def _gender(self, sentences, article, order):
        """j says what the subject is with a gendered noun ("actress"), and every other sentence about them uses pronouns."""
        text = " ".join(sentences)
        genders = [g for g, p in PRONOUNS.items() if p.search(text)]
        if len(genders) != 1 or HONORIFIC.search(text):
            return None
        gender = genders[0]
        nouns = [w for w in re.findall(r"[a-z]+", text.lower()) if w in GENDERED_WORDS or w[:-1] in GENDERED_WORDS]
        if len(nouns) != 1:
            return None
        for j in order:
            m = ROLE_GENDER.search(sentences[j])
            if not m or PRONOUNS[gender].search(sentences[j]) or (m.group("noun") in FEMALE) != (gender == "F"):
                continue
            ks = [k for k in range(len(sentences)) if k != j and PRONOUNS[gender].search(sentences[k])]
            if not 1 <= len(ks) <= 3:
                return None
            edits = {k: [Replacement(p.start(), p.end(), p.group(), swap_pronoun(p, sentences[k]), "PRONOUN")
                         for p in PRONOUNS[gender].finditer(sentences[k])] for k in ks}
            noun = m.group("noun")
            rj = [Replacement(m.start("noun"), m.end("noun"), noun, OPPOSITE[noun], "ROLE")]
            # The subject's first name takes the swapped gender too. It may occur only in j and the cascades.
            first = SUBJECT.match(sentences[0])
            if first and first.group(1) not in NOT_NAMES:
                name = first.group(1)
                holding = {i for i, s in enumerate(sentences) if word(name).search(s)}
                names = [n for n in FIRST_NAMES["F" if gender == "M" else "M"] if not word(n).search(text)]
                if not holding <= {j, *ks} or not names:
                    return None
                new = self.rng.choice(names)
                for i in holding:
                    renamed = [Replacement(n.start(), n.end(), name, new, "PERSON") for n in word(name).finditer(sentences[i])]
                    if i == j:
                        rj += renamed
                    else:
                        edits[i] += renamed
            return self._pair(sentences, j, rj, edits, f"{noun} / {'she' if gender == 'F' else 'he'}", "gender")
        return None

    def _field(self, sentences, article, order):
        """j names the subject's occupation ("physicist"), and other sentences name its field ("physics")."""
        text = " ".join(sentences)
        low = text.lower()
        for family, occupations in FIELDS.items():
            named = [o for o in occupations if re.search(rf"\b{o}s?\b", low)]
            if len(named) != 1:
                continue
            occupation = named[0]
            fields = occupations[occupation]
            for j in order:
                m = ROLE_FIELD[family].search(sentences[j])
                if not m or m.group("noun") != occupation or any(word(f).search(sentences[j].lower()) for f in fields):
                    continue
                ks = [k for k in range(len(sentences)) if k != j and any(word(f).search(sentences[k]) for f in fields)]
                if not 1 <= len(ks) <= 3:
                    break
                others = [o for o in occupations if o != occupation and not re.search(rf"\b{o}s?\b", low)
                          and not any(word(f).search(low) for f in occupations[o])]
                follows = sentences[j][:m.start("noun")].rstrip().rsplit(" ", 1)[-1].lower()
                if follows in ("a", "an"):
                    vowel = occupation[0] in "aeiou"
                    others = [o for o in others if (o[0] in "aeiou") == vowel]
                if not others:
                    break
                new = self.rng.choice(others)
                edits = {}
                for k in ks:
                    edits[k] = [Replacement(f.start(), f.end(), f.group(), occupations[new][min(i, len(occupations[new]) - 1)], "FIELD")
                                for i, field_word in enumerate(fields) for f in word(field_word).finditer(sentences[k])]
                rj = [Replacement(m.start("noun"), m.end("noun"), occupation, new, "ROLE")]
                return self._pair(sentences, j, rj, edits, f"{occupation} / {fields[0]}", "field")
        return None

    def find_and_inject_cascade(self, sentences: list[str], article: dict | None = None) -> CascadePair | None:
        """A direct edit j and a dependent edit k, or None when no strategy applies to the article."""
        if len(sentences) < 2:
            return None
        sentences = [s.strip() for s in sentences]
        order = list(range(len(sentences)))
        self.rng.shuffle(order)
        for strategy in self.rng.sample(self.strategies, len(self.strategies)):
            pair = getattr(self, f"_{strategy}")(sentences, article, order)
            if pair is not None and not fix_elsewhere(sentences, pair):
                return pair
        return None
