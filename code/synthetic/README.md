# Synthetic Data Pipeline (Stream 1)

This package implements the **Stream 1: Fast Rule-Based Factual Perturbation Pipeline** for DiffusEdit. It programmatically injects factual changes into clean paragraphs to generate 100% noise-free training and validation data, overcoming the ~40–60% label noise present in the FRUIT dataset.

## Highlights
- **Zero Label Noise**: Unchanged sentences are guaranteed `KEEP`. Stylistic edits do not occur.
- **Accurate Ground Truth Spans**: Factual entity and numeric mutations record exact character offsets (`stale_spans`) for token staleness detection.
- **Cascading Edits**: Programmatically pairs dependent edits ($s_j \rightsquigarrow s_k$) where $s_j$ has explicit evidence support and $s_k$ has downstream dependence.
- **Retractions**: Supports explicit fact retraction notifications to produce clean `DROP` supervision.
- **High Throughput**: Runs entirely on CPU in seconds.
- **Schema Compatibility**: Generated `.jsonl` files strictly comply with `classifier.inputs.build()`.

## Directory Structure
```
code/synthetic/
├── schema.py          # Data classes & validation (SyntheticRecord, EvidenceSnippet, SentenceLabel)
├── mutators/
│   ├── entities.py    # spaCy-based & regex fallback named entity mutator (PERSON, ORG, GPE, etc.)
│   ├── numbers.py     # Numeric quantities, dates, years, and measurements mutator
│   └── cascades.py    # Coupled dependencies mutator (s_j -> s_k)
├── generate.py        # CLI generator orchestrating dataset creation
├── validate.py        # Schema and token alignment integrity checker
└── README.md
```

## Quickstart

### 1. Generate Synthetic Data
From the repository root or `code/`:

**Option A: Stream diverse Wikipedia articles directly from Hugging Face (Recommended for large datasets):**
```bash
python3 -m code.synthetic.generate \
    --seed-source hf-wikipedia \
    --n 30000 \
    --out datasets/synthetic/train_30k_diverse.jsonl
```

**Option B: Generate from a local audit/seed file or default seeds:**
```bash
python3 -m code.synthetic.generate \
    --seed-file audit/train_audit.tsv \
    --n 1000 \
    --out datasets/synthetic/generated.jsonl
```

Key parameters:
- `--seed-source`: Source of seed articles: `hf-wikipedia`, `file`, or `default`.
- `--n`: Number of article records to generate (e.g. 30000).
- `--out`: Destination JSONL file path.
- `--hf-dataset`: HuggingFace dataset to stream from (default: `wikimedia/wikipedia`, fallback: `wikitext`).
- `--edit-prob`: Probability of an `EDIT` sentence (default: 0.35).
- `--drop-prob`: Probability of a `DROP` sentence (default: 0.10).
- `--cascade-prob`: Probability of injecting a cascading edit pair (default: 0.25).
- `--cascade-kinds`: Comma-separated cascade kinds to try (default: all of century, decade, location, demonym, age, duration, gender, field).
- `--require-cascade`: Edit every article and always try a cascade.
- `--seed`: Deterministic random seed (default: 42).

### 2. Validate Generated Records
Run the integrity validator:
```bash
python3 -m code.synthetic.validate datasets/synthetic/generated.jsonl
```
To run full tokenization checks against LLaDA tokenizer:
```bash
python3 -m code.synthetic.validate datasets/synthetic/generated.jsonl --check-tokens
```

### 3. Downstream Feature Extraction & Training
Directly extract features using the existing pipeline:
```bash
cd code/pipeline
python3 -m classifier.extract ../../datasets/synthetic/generated.jsonl ../../features/synthetic_train
python3 -m classifier.train --train ../../features/synthetic_train --layer 32 --out ../../heads/synthetic_sentence
python3 -m staleness.train --train ../../features/synthetic_train --val ../../features/synthetic_val --out ../../heads/synthetic_staleness
```
