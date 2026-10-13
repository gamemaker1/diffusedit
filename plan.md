# Synthetic Data Generation Plan for DiffusEdit

## 1. Executive Summary & Problem Context

The DiffusEdit pipeline relies on two critical trained heads:
1. **Sentence Classifier**: Classifies each sentence in an outdated article into `KEEP`, `EDIT`, or `DROP`.
2. **Token Staleness Detector**: Identifies token-level stale spans within `EDIT` sentences for selective re-masking and infilling.

### The Bottleneck: FRUIT Label Noise
As established in the label audit (`audit/noise.json` and Section 8.1 / Section 11 of the technical design `report.pdf` / `docs/internal/paper.tex`):
- Silver labels derived from Wikipedia revision diffs suffer from **~40–60% label noise** (KEEP: 39.8%, EDIT: 39.3%, DROP: 59.0%).
- Most Wikipedia editor revisions are **stylistic rewrites**, paraphrases, or editorial restructuring rather than factual updates supported by the attached evidence snippets.
- The heuristics classify stylistic changes as `EDIT`, causing the sentence classifier and token staleness head to overfit to stylistic rewording instead of evidence-backed factual contradiction.

### The Goal
Create a standalone, reproducible synthetic data generation pipeline in `code/synthetic/` that programmatically produces pristine, noise-free training and validation examples with:
- Exact ground-truth sentence labels (`KEEP`, `EDIT`, `DROP`).
- Precise character-level `stale_spans` for token-level staleness detection.
- Controllable evidence-text pairs that isolate factual contradiction from stylistic noise.
- Strict schema compliance with `code/pipeline/classifier/inputs.py` so the synthetic data seamlessly plugs into `classifier.extract` and `staleness.train`.

---

## 2. Model Analysis: Which Models are Used?

Based on the codebase and `report.pdf` / `docs/internal/paper.tex`:

1. **LLaDA-8B Base (`GSAI-ML/LLaDA-8B-Base`)**:
   - **Role in DiffusEdit**: The primary frozen Masked Diffusion Language Model (MDLM). It is used for:
     - Hidden state extraction (hidden dimension $d = 4096$, 32 transformer layers).
     - Token-level in-place infilling via iterative masked sampling (`code/pipeline/infilling/sampler.py` and `model.py`).
   - **Role in Synthetic Generation (Paper Section 8.2)**: Used to generate *model-refilled corruptions*. By masking a corruption fraction $t \in [0.3, 0.7]$ of a clean sentence and infilling under outdated/mismatched evidence, refilled tokens that disagree with the reference article are labeled as stale. This simulates the exact inference error distribution of the diffusion model.

2. **Entity & Linguistic Extraction Backbone (`spaCy en_core_web_sm` / `en_core_web_md`)**:
   - **Role in Synthetic Generation**: Extracts named entities (`PERSON`, `ORG`, `GPE`, `DATE`, `CARDINAL`, `EVENT`), noun chunks, and numeric values for programmatic factual perturbation.

3. **Optional Contradiction/NLI Verifier (e.g., DeBERTa-v3 / AlignScore)**:
   - Used for validating that mutated source sentences strictly contradict the synthetic evidence while unmutated sentences are entailed or unmentioned.

---

## 3. Synthetic Data Generation Methodology

To address the limitations identified in the audit and the technical design, the synthetic data pipeline will implement two complementary generation streams, combinable into a hybrid dataset:

```
┌────────────────────────────────────────────────────────┐
│             Clean Source Articles (Wikipedia)          │
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│ Stream A: Rule-Based  │       │ Stream B: LLaDA-8B    │
│ Factual Perturbation  │       │ Diffusion Infilling   │
│ (Named Entities/Nums) │       │ (Paper Section 8.2)   │
└───────────┬───────────┘       └───────────┬───────────┘
            │                               │
            ▼                               ▼
    Pristine Ground Truth           Fluency-Matched
    Sentence Labels & Spans         Model Error Distribution
            │                               │
            └───────────────┬───────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │     Schema-Compliant JSONL Output     │
        │   (Compatible with classifier/inputs) │
        └───────────────────────────────────────┘
```

### Stream A: Rule-Based Factual Perturbation (Pristine Supervision)
This stream generates unambiguous sentence-level and token-level supervision:
1. **Clean Baseline**: Start with verified, coherent Wikipedia introductory paragraphs. All sentences initially start as `KEEP` with empty `stale_spans`.
2. **Factual Evidence Generation**:
   - For an article, generate 1–5 evidence snippets containing the true, unaltered facts.
3. **Perturbation Types**:
   - **Factual `EDIT`**:
     - Identify entities using spaCy (e.g. dates, numbers, entity names, locations).
     - Perturb the entity in the source sentence (e.g. change "died in 1949" $\to$ "died in 1958", or change "won 5 titles" $\to$ "won 2 titles").
     - Record the exact character span `[start, end]` of the altered entity in `stale_spans`.
     - The corresponding evidence contains the correct original entity ("died in 1949"). The sentence is labeled `EDIT`.
   - **Factual `DROP` (Retraction)**:
     - Generate an evidence snippet that explicitly retracts or refutes a claim (e.g. "Claims that X occurred were subsequently retracted and refuted").
     - The corresponding sentence in the source text is labeled `DROP`.
   - **Pristine `KEEP`**:
     - Sentences whose facts are either supported by or irrelevant to the evidence remain untouched and are labeled `KEEP`.
   - **Cascading Edit Synthesis**:
     - If Sentence 1 changes (e.g. population: 10,000 $\to$ 50,000), update Sentence 2 which references the dependent fact (e.g. "As a small town under 15,000 residents..."). Sentence 1 is a direct edit, and Sentence 2 is a synthetic cascading edit.

### Stream B: Diffusion-Based Infilling Corruptions (Paper Sec 8.2 / Mask-Disc)
1. Sample clean Wikipedia articles.
2. Sample corruption fraction $t \sim \mathcal{U}[0.3, 0.7]$.
3. Mask $t \times L$ tokens in chosen sentences using LLaDA's `MASK_ID` (`126336`).
4. Run frozen LLaDA-8B infilling sampler (`code/pipeline/infilling/sampler.py`) conditioned on evidence.
5. Diff the generated tokens against the original clean tokens. Any generated token differing from the gold original is labeled stale (`stale = 1`), and the sentence is labeled `EDIT`.
6. This equips the staleness detector to distinguish subtle model hallucinations from supported facts.

---

## 4. Pipeline Architecture & Directory Structure

All synthetic generation code will be organized in `code/synthetic/`:

```
code/synthetic/
├── README.md               # Documentation and quickstart usage
├── schema.py               # Data models and JSONL validation (matches classifier.inputs)
├── mutators/
│   ├── __init__.py
│   ├── entities.py         # Named entity & mention perturbation (spaCy)
│   ├── numbers.py          # Numeric, dates, and measurement swappers
│   └── cascades.py         # Secondary / consequential dependency mutators
├── infill_corrupt.py       # LLaDA-8B diffusion infilling corruption (Paper Sec 8.2)
├── generate.py             # CLI runner orchestrating dataset generation
└── validate.py             # Integrity tests checking token alignments & span offsets
```

### JSONL Output Schema
The generated synthetic dataset strictly matches `code/pipeline/classifier/inputs.py`:
```json
{
  "id": "synthetic-000123",
  "evidence": [
    {
      "title": "Document Title",
      "text": "Evidence snippet containing factual context."
    }
  ],
  "source_sentences": [
    "First sentence that remains completely unchanged.",
    "Second sentence with a perturbed entity.",
    "Third sentence that was retracted."
  ],
  "labels": [
    {
      "label": "KEEP",
      "stale_spans": []
    },
    {
      "label": "EDIT",
      "stale_spans": [[23, 39]]
    },
    {
      "label": "DROP",
      "stale_spans": []
    }
  ]
}
```

---

## 5. Execution Steps

1. **Phase 1: Implementation of Core Generators (`code/synthetic/`)**
   - Create schema validation and data classes (`schema.py`).
   - Implement entity, numeric, and date mutators (`mutators/`).
   - Implement infilling-based corruption using LLaDA-8B (`infill_corrupt.py`).
   - Implement the CLI pipeline (`generate.py`).

2. **Phase 2: Validation & Verification**
   - Generate a test batch of 100 synthetic instances.
   - Run `python -m classifier.inputs <synthetic.jsonl> --n 50` to verify:
     - Exact token span alignment.
     - Accurate mapping of `stale_spans` to token-level staleness masks.
     - Zero mismatches between decoded tokens and source sentences.

3. **Phase 3: Integration with Feature Extraction and Training**
   - Run `classifier.extract` on the synthetic data.
   - Run `classifier.train` and `staleness.train` on:
     - Pure synthetic data.
     - Mixed synthetic + silver FRUIT data (ablation axis in Table 4 of paper).
   - Evaluate against the audited gold split (`audit/train_audit.tsv`).
