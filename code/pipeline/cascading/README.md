# Cascading Edit Detection

This package implements the cascade detection system from Sections 6.4, 7.3 and 8.5 of the DiffusEdit technical design.

## What is a Cascading Edit?

When an article says *"Three people died and the small death toll limited the political fallout"*, and the true count is updated to 42, the first sentence is directly contradicted by evidence. The second sentence is wrong in a quieter way — it rests on the now-false premise of a small toll. An editor who fixes the first sentence and stops has left the article internally inconsistent. The second sentence is a **cascading edit**.

## Module Structure

```
cascading/
├── mentions.py    M(text): named entity + digit string extractor (Section 7.3)
├── detectors.py   Three cascade detectors: (a) mention overlap, (b) staleness delta, (c) consequence prediction
├── train.py       ConsequenceHead training — Equation 8
├── mine.py        Cascade pair mining from silver training data (Section 8.5)
└── evaluate.py    Cascade recall evaluation on hand-labeled pairs (Section 9.2)
```

## Three Detectors

### (a) Mention Overlap — `MentionOverlapDetector`
No training. Flags sentence `s_k` when its mention set `M(s_k)` shares at least one string with the mentions of any sentence repaired this round. High recall, low precision baseline.

```bash
# No training needed — use directly in infilling.run
```

### (b) Staleness Delta — `StalenessDeltaDetector`
No training. Re-scores every sentence under the fresh hidden states (after repairs) and flags sentences whose staleness score rose into the top 10th percentile and crosses `τ_2`. Requires the existing staleness head.

```bash
# No training needed — requires --staleness-head from staleness.train
```

### (c) Consequence Prediction — `ConsequenceDetector`
Trained head (Equation 8). For each repaired sentence `s_j` and each unrepaired sentence `s_k`:

```
g_jk = σ( MLP_ψ( [u_j ; u_k ; u_j ⊙ u_k ; |u_j − u_k|] ) )
```

`u_j` is the LLaDA pooled vector of `s_j` in its **corrected** version; `u_k` is `s_k` in its **outdated** version. Flags `s_k` if `g_jk ≥ τ_c`.

## Training the Consequence Head

### Step 1 — Mine Training Pairs (Section 8.5)

```bash
python -m cascading.mine \
    --data datasets/collated/train.jsonl \
    --features features/train \
    --out pairs/cascade_pairs.jsonl \
    --threshold 0.6
```

This produces `pairs/cascade_pairs.jsonl` with positive (cascade) and negative (keep) pairs. Before training, hand-label 200 mined pairs. If fewer than ~60% are genuine cascades, drop consequence prediction and rely on detectors (a) and (b).

### Step 2 — Train the Head

```bash
python -m cascading.train \
    --features features/train \
    --pairs pairs/cascade_pairs.jsonl \
    --out heads/cascade
```

### Step 3 — Evaluate Cascade Recall (Section 9.2)

```bash
# Detector (a): mention overlap
python -m cascading.evaluate \
    --data datasets/collated/train.jsonl \
    --labels pairs/eval_labels.jsonl \
    --features features/train \
    --detector mention \
    --out eval/cascade_mention.json

# Detector (c): consequence prediction
python -m cascading.evaluate \
    --data datasets/collated/train.jsonl \
    --labels pairs/eval_labels.jsonl \
    --features features/train \
    --detector consequence \
    --cascade-head heads/cascade \
    --tau-c 0.5 \
    --out eval/cascade_consequence.json
```

Results are broken into:
- **Group A**: cascades sharing a mention with a direct edit (all three detectors can reach these)
- **Group B**: cascades without shared mention (only Detector (c) can reach these)

Detector (a) scores 0 on Group B by construction — this is the gap Detector (c) is designed to close.

## Integration with `infilling/run.py`

The cascade module plugs into **Algorithm 1, line 8** of `infilling/run.py`:

```python
from cascading.detectors import MentionOverlapDetector

detector = MentionOverlapDetector()

# After each infilling round:
flagged = detector.flag(
    sentences=current_sentences,
    delta_r=repaired_this_round,      # set[int] of sentence indices
    pooled_before=H_before,           # [n_sents, d] from previous encode
    pooled_after=H_after,             # [n_sents, d] from line-7 re-encode
)
# flagged sentences enter the next round as EDIT
```

## Expected Performance (from the paper's framing)

| Detector | Group A recall | Group B recall | Training needed |
|---|---|---|---|
| (a) Mention overlap | High | 0% by construction | None |
| (b) Staleness delta | Medium | Medium | None (uses staleness head) |
| (c) Consequence pred | High | Non-zero | Yes — `cascading.train` |
