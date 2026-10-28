"""Stream 1: Fast, rule-based synthetic data generator for DiffusEdit.

Generates 100% noise-free synthetic training examples for sentence classification
(KEEP/EDIT/DROP) and token staleness detection (stale_spans).

Features:
- Entity swaps drawn from the seed corpus itself, of the same type and shape,
  often from the same article, and number and date changes of varied size.
- One to three changed spans per EDIT sentence, and per-article edit rates.
- Cascading edits (s_j -> s_k) of the kinds in synthetic/mutators/cascades.py,
  with evidence for s_j only. Each link records its kind.
- Evidence in varied forms (synthetic/evidence.py): restatements, excerpts,
  fact-sheet fields, retractions, confirmations of KEEP sentences in the same
  frames, and snippets from other articles.
- Pristine KEEP sentences with zero stylistic changes.
- Direct JSONL output matching classifier.inputs.build() requirements.

Usage:
    python -m synthetic.generate --seed-file datasets/synthetic/split/train.jsonl --n 5000 \
        --cascade-prob 1.0 --prefix cascade- --out datasets/synthetic/cascade/source.jsonl
"""

import argparse
import json
from pathlib import Path
import random
import re
import sys

# Ensure both package and direct script execution work cleanly
_curr = Path(__file__).resolve().parent
if str(_curr.parent) not in sys.path:
    sys.path.insert(0, str(_curr.parent))
if str(_curr) not in sys.path:
    sys.path.insert(0, str(_curr))

try:
    from .evidence import Writer
    from .mutators import CascadeMutator, EntityMutator, NumberMutator
    from .mutators.entities import EntityBank, apply
    from .schema import EvidenceSnippet, SentenceLabel, SyntheticRecord, write_jsonl
except ImportError:
    from evidence import Writer
    from mutators import CascadeMutator, EntityMutator, NumberMutator
    from mutators.entities import EntityBank, apply
    from schema import EvidenceSnippet, SentenceLabel, SyntheticRecord, write_jsonl


# Seed corpus of factual, diverse reference paragraphs
DEFAULT_SEEDS = [
    [
        "Raimond Valgre was an Estonian composer and musician whose songs became widely recognized in Estonia.",
        "During World War II, he was a member of the orchestra of the 8th Estonian Rifle Corps.",
        "His music was officially banned in 1948 by Soviet authorities.",
        "Valgre died in an accident on 31 December 1949.",
        "The revival of Valgre's compositions began in the Soviet Union during the 1960s.",
        "In 2001, Francis Goya recorded twelve of his songs in an international album.",
    ],
    [
        "Brooklyn Borough Hall was designed by architects Calvin Pollard and Gamaliel King in the Greek Revival style.",
        "It was completed in 1848 to serve as the City Hall of Brooklyn.",
        "In January 1898, the City of Brooklyn merged with New York City, and the building became Brooklyn Borough Hall.",
        "The building features exterior Tuckahoe marble and an ornate cast-iron cupola.",
        "Major restoration of the building facade took place in 1989.",
    ],
    [
        "The Zaza Formation is a palaeontological formation located in Buryatia, Russia.",
        "It dates to the Lower Cretaceous period, approximately 125 million years ago.",
        "Fossil insects, plants, and freshwater molluscs have been recovered from the lacustrine deposits.",
        "The formation preserves some of the earliest known flowering plants in Eastern Asia.",
    ],
    [
        "Alexander Wright founded Apex Global Logistics in Toronto in 1994.",
        "The company expanded operations across North America with over 5,000 employees.",
        "In 2012, the firm invested $50 million in automated distribution centers.",
        "The headquarters relocated to Montreal following the acquisition of Boreal Financial Group in 2018.",
        "Revenue exceeded $1.2 billion in the fiscal year ending 2022.",
    ],
    [
        "The Treaty of Portsmouth was signed on 5 September 1905, formally ending the Russo-Japanese War.",
        "Negotiations were brokered by President Theodore Roosevelt at the Portsmouth Naval Shipyard in Maine.",
        "Roosevelt was awarded the Nobel Peace Prize in 1906 for his mediation efforts.",
        "The agreement recognized Japan's sphere of influence in Korea and southern Manchuria.",
        "Both nations agreed to evacuate military forces from Manchuria within eighteen months.",
    ],
    [
        "Elena Rostova joined the Kyoto Institute of Technology as lead researcher in 2015.",
        "Her team developed novel photovoltaic materials with an energy efficiency of 28 percent.",
        "The findings were published in Nature Materials in March 2019.",
        "A pilot manufacturing plant was opened in Osaka in 2021 to test commercial scalability.",
        "The institute received a grant of 15 million yen from the Ministry of Education.",
    ],
    [
        "The Great Northern Railway was constructed across the northern United States in the late nineteenth century.",
        "Completed in 1893, it was built without federal land grants under the leadership of James J. Hill.",
        "The route stretched from Saint Paul, Minnesota to Seattle, Washington.",
        "It promoted settlement along the line by offering low fares to prospective farmers.",
        "The company merged into the Burlington Northern Railroad in 1970.",
    ],
]


class SyntheticGenerator:
    """Outdated articles with exact labels, made from correct seed articles.

    Each article draws its own edit and drop rates around the global ones, and
    a share P_UNCHANGED of articles keeps every sentence. An EDIT sentence
    changes one to three entities, numbers or dates (SPANS_PER_EDIT). The
    evidence states every EDIT fix, retracts every DROP sentence, restates a
    share P_SUPPORT of KEEP sentences, confirms a share P_DECOY of them in the
    frame of a retraction, and adds up to two snippets from other articles.
    Snippets are merged in pairs with probability P_MERGE and shuffled, and
    the optional ones are removed first when the evidence exceeds
    EVIDENCE_CHARS.
    """

    P_UNCHANGED = 0.08
    P_SUPPORT = 0.35
    P_DECOY = 0.1
    P_MERGE = 0.2
    SPANS_PER_EDIT = ((1, 0.6), (2, 0.3), (3, 0.1))
    DISTRACTORS = (0, 0, 1, 1, 2)
    EVIDENCE_CHARS = 1800

    def __init__(
        self,
        seeds: list[list[str]] | None = None,
        seed: int = 42,
        edit_prob: float = 0.35,
        drop_prob: float = 0.10,
        cascade_prob: float = 0.20,
        cascade_kinds=None,
        require_cascade: bool = False,
    ):
        self.rng = random.Random(seed)
        self.seeds = [[s.strip() for s in article] for article in (seeds or DEFAULT_SEEDS)]
        self.edit_prob = edit_prob
        self.drop_prob = drop_prob
        self.cascade_prob = cascade_prob
        # With require_cascade every article is edited and tries a cascade.
        self.require_cascade = require_cascade

        self.bank = EntityBank(self.seeds)
        self.entity_mutator = EntityMutator(seed=seed, bank=self.bank)
        self.number_mutator = NumberMutator(seed=seed + 1)
        self.cascade_mutator = CascadeMutator(seed=seed + 2, bank=self.bank, **({"strategies": cascade_kinds} if cascade_kinds else {}))
        self.writer = Writer(random.Random(seed + 3))

    def _edit(self, sentence, article):
        """(mutated sentence, stale spans, fixed spans of the correct sentence), or None."""
        options = self.entity_mutator.candidates(sentence, article) + self.number_mutator.candidates(sentence)
        if not options:
            return None
        self.rng.shuffle(options)
        k = self.rng.choices([n for n, _ in self.SPANS_PER_EDIT], weights=[w for _, w in self.SPANS_PER_EDIT])[0]
        chosen = []
        for r in options:
            if len(chosen) < k and not any(r.start < c.end and r.end > c.start for c in chosen):
                chosen.append(r)
        mutated, spans = apply(sentence, chosen)
        return mutated, spans, sorted((r.start, r.end) for r in chosen)

    def _main_entity(self, article_sentences):
        counts = {}
        for s in article_sentences:
            for _, _, text, label in self.bank.entities(s):
                if label in ("PERSON", "ORG", "GPE", "FAC", "EVENT", "WORK_OF_ART", "LOC"):
                    counts[text] = counts.get(text, 0) + 1
        return max(sorted(counts), key=counts.get) if counts else None

    def generate_one(
        self, record_id: str, base_sentences: list[str] | None = None, seed_index: int | None = None
    ) -> SyntheticRecord:
        """Generate a single synthetic article record."""
        if base_sentences is None:
            seed_index = self.rng.randrange(len(self.seeds))
            base_sentences = self.seeds[seed_index]
        base = [s.strip() for s in base_sentences]
        if seed_index is not None and seed_index < len(self.bank.article_entities):
            article = self.bank.article_entities[seed_index]
        else:
            article = {}
            for s in base:
                for _, _, text, label in self.bank.entities(s):
                    article.setdefault(label, set()).add(text)

        if self.rng.random() < self.P_UNCHANGED and not self.require_cascade:
            edit_prob = drop_prob = 0.0
        else:
            edit_prob = min(0.9, self.edit_prob * self.rng.uniform(0.3, 1.7))
            drop_prob = min(0.5, self.drop_prob * self.rng.uniform(0.0, 2.0))

        required, optional = [], []

        def snippet(text, kind, about=base):
            year = next((m.group() for s in about for m in re.finditer(r"\b(1[5-9]\d\d|20[0-2]\d)\b", s)), None)
            return {"title": self.writer.title(self._main_entity(about), year), "text": text, "kind": kind, "about": about}

        cascade_pair = None
        if edit_prob and (self.require_cascade or self.rng.random() < self.cascade_prob) and len(base) >= 2:
            cascade_pair = self.cascade_mutator.find_and_inject_cascade(base, article)

        source_sentences, labels = [], []
        for i, sentence in enumerate(base):
            if cascade_pair and i == cascade_pair.direct_sentence_idx:
                source_sentences.append(cascade_pair.direct_mutated)
                labels.append(SentenceLabel("EDIT", cascade_pair.direct_stale_spans, sentence))
                focus = [(m.start(), m.end()) for m in re.finditer(re.escape(cascade_pair.shared_concept.split(" / ")[0]), sentence)]
                required.append(snippet(self.writer.support(sentence, focus[:1] or None), "support"))
                continue
            if cascade_pair and i in cascade_pair.cascade_edits:
                mutated, spans = cascade_pair.cascade_edits[i]
                source_sentences.append(mutated)
                labels.append(SentenceLabel("EDIT", spans, sentence))
                continue

            r = self.rng.random()
            if r < drop_prob:
                source_sentences.append(sentence)
                labels.append(SentenceLabel("DROP", [], ""))
                required.append(snippet(self.writer.notice(sentence, retract=True), "notice"))
                continue
            if r < drop_prob + edit_prob:
                edit = self._edit(sentence, article)
                if edit is not None:
                    mutated, spans, focus = edit
                    source_sentences.append(mutated)
                    labels.append(SentenceLabel("EDIT", spans, sentence))
                    required.append(snippet(self.writer.support(sentence, focus), "support"))
                    continue

            source_sentences.append(sentence)
            labels.append(SentenceLabel("KEEP", [], sentence))
            if self.rng.random() < self.P_SUPPORT:
                optional.append(snippet(self.writer.support(sentence), "support"))
            elif self.rng.random() < self.P_DECOY:
                optional.append(snippet(self.writer.notice(sentence, retract=False), "notice"))

        for _ in range(self.rng.choice(self.DISTRACTORS)):
            other = self.rng.choice(self.seeds)
            if other is not base_sentences:
                optional.append(snippet(self.writer.support(self.rng.choice(other)), "support", other))

        if cascade_pair:
            # No snippet may state the cascade's fix, in its title or its text.
            patterns = [re.compile(rf"(?<!\w){re.escape(f.lower())}(?!\w)") for f in cascade_pair.cascade_fixes]
            def leaks(s):
                return any(p.search(f"{s['title']} {s['text']}".lower()) for p in patterns)
            for s in required:
                if leaks(s):
                    # A title without an entity or year holds no fix string. A
                    # required text (a retraction quoting a DROP sentence) stays.
                    s["title"] = self.writer.title(None, None)
            optional = [s for s in optional if not leaks(s)]
        self.rng.shuffle(optional)
        while optional and sum(len(s["title"]) + len(s["text"]) for s in required + optional) > self.EVIDENCE_CHARS:
            optional.pop()
        snippets = required + optional
        self.rng.shuffle(snippets)
        merged = []
        for s in snippets:
            if (merged and s["kind"] == merged[-1]["kind"] == "support" and s["about"] is merged[-1]["about"]
                    and self.rng.random() < self.P_MERGE):
                merged[-1] = {**merged[-1], "text": f"{merged[-1]['text']} {s['text']}", "kind": "merged"}
            else:
                merged.append(s)
        if not merged:
            other = self.rng.choice(self.seeds)
            merged.append(snippet(self.writer.support(self.rng.choice(other)), "support", other))
        evidence = [EvidenceSnippet(s["title"], s["text"]) for s in merged]

        cascades = []
        if cascade_pair:
            cascades = [{"direct": cascade_pair.direct_sentence_idx, "cascade": k, "kind": cascade_pair.strategy}
                        for k in cascade_pair.cascade_edits]
        return SyntheticRecord(
            id=record_id,
            evidence=evidence,
            source_sentences=source_sentences,
            labels=labels,
            target_sentences=[l.target for l in labels if l.label != "DROP"],
            cascades=cascades,
        )

    def generate_dataset(self, n_records: int, prefix: str = "synthetic-") -> list[SyntheticRecord]:
        """Generate N synthetic article records.

        Seeds are drawn without replacement in a shuffled order, so no two
        records share a seed until every seed has been used once.
        """
        records, order = [], []
        for i in range(n_records):
            if not order:
                order = list(range(len(self.seeds)))
                self.rng.shuffle(order)
            index = order.pop()
            records.append(self.generate_one(f"{prefix}{i:06d}", self.seeds[index], index))
        return records


def load_seeds_from_file(path: str | Path) -> list[list[str]]:
    """Load seed paragraphs from a TSV, JSONL, or text file."""
    path = Path(path)
    seeds = []
    if path.suffix == ".tsv":
        # Group sentences by article ID (e.g. audit/train_audit.tsv)
        articles: dict[str, list[str]] = {}
        with path.open(encoding="utf-8") as f:
            headers = [h.strip() for h in f.readline().split("\t")]
            id_idx = headers.index("id") if "id" in headers else 0
            src_idx = headers.index("source") if "source" in headers else 4
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > src_idx:
                    art_id = parts[id_idx]
                    src_sent = parts[src_idx].strip()
                    if src_sent:
                        articles.setdefault(art_id, []).append(src_sent)
        seeds = [sents for sents in articles.values() if len(sents) >= 2]
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    # A synthetic record's gold article is its target_sentences.
                    # Its source sentences hold the injected errors.
                    target = data.get("target_sentences", [])
                    sents = target if target and isinstance(target[0], str) else data.get("source_sentences", [])
                    if len(sents) >= 2:
                        seeds.append(sents)
    else:
        text = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p in paragraphs:
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
            if len(sents) >= 2:
                seeds.append(sents)
    return seeds


def stream_hf_wikipedia(
    n_articles: int,
    dataset_name: str = "wikimedia/wikipedia",
    config: str = "20231101.en",
):
    """Stream clean Wikipedia lead paragraphs from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for --seed-source hf-wikipedia.\n"
            "Install it via: uv add datasets  (or pip install datasets)"
        )

    print(f"Connecting to Hugging Face dataset '{dataset_name}' (config: {config}, streaming=True)...")
    try:
        ds = load_dataset(dataset_name, config, split="train", streaming=True)
    except Exception as e:
        print(f"Notice: Failed to load '{dataset_name}' ({e}). Falling back to 'wikitext' (wikitext-103-raw-v1)...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

    count = 0
    for row in ds:
        text = row.get("text", "")
        if not text:
            continue

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        lead = None
        for p in paragraphs:
            # Strip wiki headers and markup
            cleaned = re.sub(r"=+\s*[^=]+\s*=+", "", p).strip()
            cleaned = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", cleaned)
            if (
                len(cleaned) > 120
                and not cleaned.startswith(("File:", "Image:", "thumb|", "{|"))
                and not cleaned.endswith(":")
            ):
                lead = cleaned
                break

        if not lead:
            continue

        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", lead) if s.strip()]
        # Filter for solid article introductory paragraphs with 2 to 8 sentences
        if 2 <= len(sents) <= 10:
            yield sents[:8]
            count += 1
            if count >= n_articles:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=500, help="number of records to generate")
    parser.add_argument("--out", type=str, required=True, help="output jsonl path")
    parser.add_argument(
        "--seed-source",
        choices=["file", "hf-wikipedia", "default"],
        default="default",
        help="source of seed articles: file, hf-wikipedia, or default",
    )
    parser.add_argument("--seed-file", type=str, default=None, help="path to TSV, JSONL, or text file with seed Wikipedia paragraphs")
    parser.add_argument("--hf-dataset", type=str, default="wikimedia/wikipedia", help="Hugging Face dataset identifier")
    parser.add_argument("--hf-config", type=str, default="20231101.en", help="Hugging Face dataset config")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--prefix", type=str, default="synthetic-", help="prefix of the record ids")
    parser.add_argument("--edit-prob", type=float, default=0.35, help="mean probability of EDIT per sentence")
    parser.add_argument("--drop-prob", type=float, default=0.10, help="mean probability of DROP per sentence")
    parser.add_argument("--cascade-prob", type=float, default=0.25, help="probability of cascade injection")
    parser.add_argument("--cascade-kinds", help="comma-separated cascade kinds to try (default: all)")
    parser.add_argument("--require-cascade", action="store_true", help="edit every article and always try a cascade")
    args = parser.parse_args()

    if args.seed_file:
        args.seed_source = "file"
    seeds = None
    if args.seed_source == "file":
        seeds = load_seeds_from_file(args.seed_file)
        print(f"Loaded {len(seeds)} seed articles from {args.seed_file}")
    elif args.seed_source == "hf-wikipedia":
        seeds = list(stream_hf_wikipedia(args.n, dataset_name=args.hf_dataset, config=args.hf_config))
        print(f"Streamed {len(seeds)} seed articles from {args.hf_dataset}")

    print("Indexing the entities of the seed articles...")
    generator = SyntheticGenerator(
        seeds=seeds,
        seed=args.seed,
        edit_prob=args.edit_prob,
        drop_prob=args.drop_prob,
        cascade_prob=args.cascade_prob,
        cascade_kinds=args.cascade_kinds.split(",") if args.cascade_kinds else None,
        require_cascade=args.require_cascade,
    )
    print(f"Generating {args.n} synthetic records with seed {args.seed}...")
    records = generator.generate_dataset(args.n, prefix=args.prefix)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, args.out)

    label_counts = {"KEEP": 0, "EDIT": 0, "DROP": 0}
    total_sentences = sum(len(r.source_sentences) for r in records)
    total_stale_spans = 0
    for r in records:
        for lbl in r.labels:
            label_counts[lbl.label] += 1
            total_stale_spans += len(lbl.stale_spans)
    print(f"Saved {len(records)} records to {args.out}")
    print(f"Total sentences: {total_sentences}")
    if total_sentences > 0:
        for k, v in label_counts.items():
            print(f"  {k}: {v} ({v / total_sentences:.1%})")
    print(f"Total stale spans generated: {total_stale_spans}")
    print(f"Records with a cascade link: {sum(bool(r.cascades) for r in records)}")


if __name__ == "__main__":
    main()
