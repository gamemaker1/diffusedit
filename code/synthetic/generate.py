"""Stream 1: Fast, rule-based synthetic data generator for DiffusEdit.

Generates 100% noise-free synthetic training examples for sentence classification
(KEEP/EDIT/DROP) and token staleness detection (stale_spans).

Features:
- Programmatic entity, date, and numeric perturbations.
- Cascading edit pairs (s_j -> s_k) where s_j has direct evidence and s_k is downstream.
- Retraction-based DROP sentences supported by explicit evidence.
- Pristine KEEP sentences with zero stylistic changes.
- Direct JSONL output matching classifier.inputs.build() requirements.

Usage:
    python -m code.synthetic.generate --n 1000 --out datasets/synthetic/train.jsonl
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
    from .mutators import CascadeMutator, EntityMutator, NumberMutator
    from .schema import EvidenceSnippet, SentenceLabel, SyntheticRecord, write_jsonl
except ImportError:
    from mutators import CascadeMutator, EntityMutator, NumberMutator
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
    def __init__(
        self,
        seeds: list[list[str]] | None = None,
        seed: int = 42,
        edit_prob: float = 0.35,
        drop_prob: float = 0.10,
        cascade_prob: float = 0.20,
    ):
        self.rng = random.Random(seed)
        self.seeds = seeds or DEFAULT_SEEDS
        self.edit_prob = edit_prob
        self.drop_prob = drop_prob
        self.cascade_prob = cascade_prob

        self.entity_mutator = EntityMutator(seed=seed)
        self.number_mutator = NumberMutator(seed=seed + 1)
        self.cascade_mutator = CascadeMutator(seed=seed + 2)

    def generate_one(
        self, record_id: str, base_sentences: list[str] | None = None
    ) -> SyntheticRecord:
        """Generate a single synthetic article record."""
        # Pick a base article
        if base_sentences is None:
            base_sentences = list(self.rng.choice(self.seeds))
        else:
            base_sentences = list(base_sentences)
        n_sentences = len(base_sentences)

        source_sentences = []
        labels = []
        evidence = []

        # Decide whether to inject a cascading edit pair
        cascade_pair = None
        if self.rng.random() < self.cascade_prob and n_sentences >= 3:
            cascade_pair = self.cascade_mutator.find_and_inject_cascade(base_sentences)

        if cascade_pair:
            # Evidence only contains information for the direct edit, not the cascade!
            evidence.append(
                EvidenceSnippet(
                    title="Verified Update Briefing",
                    text=f"Confirmed reporting states: {cascade_pair.direct_original}",
                )
            )

        for i, original_sent in enumerate(base_sentences):
            sent_clean = original_sent.strip()

            # Handle cascade pair assignments
            if cascade_pair and i == cascade_pair.direct_sentence_idx:
                source_sentences.append(cascade_pair.direct_mutated)
                labels.append(SentenceLabel(
                    label="EDIT",
                    stale_spans=cascade_pair.direct_stale_spans,
                    target=cascade_pair.direct_original,
                ))
                continue
            elif cascade_pair and i == cascade_pair.cascade_sentence_idx:
                source_sentences.append(cascade_pair.cascade_mutated)
                labels.append(SentenceLabel(
                    label="EDIT",
                    stale_spans=cascade_pair.cascade_stale_spans,
                    target=cascade_pair.cascade_original,
                ))
                continue

            r = self.rng.random()

            # 1. DROP (Retraction supported by explicit evidence)
            if r < self.drop_prob:
                evidence.append(
                    EvidenceSnippet(
                        title="Correction Notice",
                        text=f"The claim that '{sent_clean}' was retracted following audit.",
                    )
                )
                source_sentences.append(sent_clean)
                labels.append(SentenceLabel(label="DROP", stale_spans=[], target=""))
                continue

            # 2. EDIT (Factual corruption backed by original fact in evidence)
            elif r < (self.drop_prob + self.edit_prob):
                # Randomly attempt entity or number mutation
                mut = None
                if self.rng.random() < 0.5:
                    mut = self.entity_mutator.mutate(sent_clean) or self.number_mutator.mutate(sent_clean)
                else:
                    mut = self.number_mutator.mutate(sent_clean) or self.entity_mutator.mutate(sent_clean)

                if mut is not None:
                    # The source has the outdated/mutated text
                    source_sentences.append(mut.mutated_text)
                    labels.append(SentenceLabel(label="EDIT", stale_spans=mut.stale_spans, target=sent_clean))
                    # Evidence gives the true original fact
                    evidence.append(
                        EvidenceSnippet(
                            title="Reference Fact Sheet",
                            text=f"Verified records document: {sent_clean}",
                        )
                    )
                    continue

            # 3. KEEP (Unchanged, 0 noise)
            source_sentences.append(sent_clean)
            labels.append(SentenceLabel(label="KEEP", stale_spans=[], target=sent_clean))

        # Ensure at least one evidence snippet exists
        if not evidence:
            evidence.append(
                EvidenceSnippet(
                    title="General Reference Document",
                    text="Archival overview covering confirmed historical background.",
                )
            )

        target_sentences = [lbl.target for lbl in labels if lbl.label != "DROP"]

        return SyntheticRecord(
            id=record_id,
            evidence=evidence,
            source_sentences=source_sentences,
            labels=labels,
            target_sentences=target_sentences,
        )

    def generate_dataset(self, n_records: int, prefix: str = "synthetic-") -> list[SyntheticRecord]:
        """Generate N synthetic article records."""
        records = []
        for i in range(n_records):
            rec_id = f"{prefix}{i:06d}"
            records.append(self.generate_one(rec_id))
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
                    sents = data.get("source_sentences", [])
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
    parser.add_argument("--out", type=str, default="datasets/synthetic/train.jsonl", help="output jsonl path")
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
    parser.add_argument("--edit-prob", type=float, default=0.35, help="probability of EDIT per sentence")
    parser.add_argument("--drop-prob", type=float, default=0.10, help="probability of DROP per sentence")
    parser.add_argument("--cascade-prob", type=float, default=0.25, help="probability of cascade injection")
    args = parser.parse_args()

    # Automatically set seed-source to file if seed-file is passed
    if args.seed_file:
        args.seed_source = "file"

    generator = SyntheticGenerator(
        seed=args.seed,
        edit_prob=args.edit_prob,
        drop_prob=args.drop_prob,
        cascade_prob=args.cascade_prob,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    label_counts = {"KEEP": 0, "EDIT": 0, "DROP": 0}
    total_stale_spans = 0
    total_sentences = 0
    record_count = 0

    if args.seed_source == "hf-wikipedia":
        print(f"Generating {args.n} synthetic records from streamed Hugging Face Wikipedia articles...")
        with open(args.out, "w", encoding="utf-8") as f_out:
            for article_sents in stream_hf_wikipedia(args.n, dataset_name=args.hf_dataset, config=args.hf_config):
                rec_id = f"synthetic-{record_count:06d}"
                rec = generator.generate_one(rec_id, base_sentences=article_sents)
                f_out.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

                total_sentences += len(rec.source_sentences)
                for lbl in rec.labels:
                    label_counts[lbl.label] += 1
                    total_stale_spans += len(lbl.stale_spans)
                record_count += 1
                if record_count % 1000 == 0:
                    print(f"  Progress: generated {record_count} / {args.n} records...")
    else:
        seeds = None
        if args.seed_source == "file" and args.seed_file:
            seeds = load_seeds_from_file(args.seed_file)
            print(f"Loaded {len(seeds)} seed articles from {args.seed_file}")
            generator.seeds = seeds

        print(f"Generating {args.n} synthetic records with seed {args.seed}...")
        records = generator.generate_dataset(args.n)
        write_jsonl(records, args.out)
        record_count = len(records)
        total_sentences = sum(len(r.source_sentences) for r in records)
        for r in records:
            for lbl in r.labels:
                label_counts[lbl.label] += 1
                total_stale_spans += len(lbl.stale_spans)

    print(f"Saved {record_count} records to {args.out}")
    print(f"Total sentences: {total_sentences}")
    if total_sentences > 0:
        for k, v in label_counts.items():
            print(f"  {k}: {v} ({v / total_sentences:.1%})")
    print(f"Total stale spans generated: {total_stale_spans}")


if __name__ == "__main__":
    main()
