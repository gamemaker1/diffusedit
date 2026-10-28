# Cascading Edit Detection

> Detectors that flag sentences made wrong by the repair of another sentence (Sections 6.4, 7.3 and 8.5 of the technical design).

A cascading edit is a sentence that the evidence does not contradict but that depends on a sentence the evidence does contradict. In "Three people died, and the small death toll limited the political fallout", an update of the count to 42 contradicts the first clause, and the second clause becomes wrong only because of it. A round of Algorithm 1 repairs the direct edits, and a detector then flags the sentences outside the repaired set Δ that the next round treats as EDIT.

## Detectors

| Detector | Class | Inputs | Training |
|---|---|---|---|
| (a) Mention overlap | `MentionOverlapDetector` | sentences before and after the round | none |
| (b) Staleness delta | `StalenessDeltaDetector` | staleness head over the hidden states before and after the round | none |
| (c) Consequence prediction | `ConsequenceDetector` | consequence head over u_j (corrected) and u_k (outdated) | `cascading.train` |
| (d) Edited as evidence | `EditedAsEvidenceDetector` | classifier head over the evidence with the repairs added | none |

The letters (a) to (c) are the paper's, and edited as evidence, from the progress report, is (d). Mention overlap flags a sentence that shares a named entity or digit string with either version of a repaired sentence, so it cannot reach a cascade without a shared mention. Staleness delta flags a sentence with a token that enters the top tenth of the article's token scores after the round, was not there before, and scores at least `tau2`. In `infilling.loop` it re-scores every token with the masking head. The offline evaluation has no token states for every sentence and applies the same rule to one score per sentence from its pooled vector. Consequence prediction flags `s_k` when `g_jk >= tau_c` for some repaired `s_j`, with `tau_c` tuned for F1 on the validation pairs by default. Edited as evidence puts the repaired sentences before the other snippets, so the evidence token budget never cuts them.

## Data

The synthetic set records each injected cascade as a `{"direct": j, "cascade": k, "kind": ...}` link in the `cascades` field of the record, so labels and training pairs come from these links and no mining or hand labelling is needed. In each link, the correct string of k differs from the fix of j, and the evidence states only j's fix, so k can be repaired only by reasoning from j. The kinds are century, decade, location (a city and its country), demonym, age, duration, gender (a gendered role noun and the subject's pronouns) and field (an occupation and its field), as described in `synthetic/mutators/cascades.py`. One j of the gender and field kinds may have several cascades. Commands run from `code/pipeline`.

```bash
D=../datasets/synthetic/split
python -m cascading.links $D/train.jsonl $D --heads-holdout 0.1
python -m cascading.views $D/train.jsonl $D
```

`cascading.links` leaves out a link whose cascade fix also occurs in another sentence of the outdated article. With `--heads-holdout 0.1`, the evaluation articles in `labels.jsonl` are the linked articles among those the classifier and staleness heads hold out, and the other linked articles give `pairs.jsonl`. Each cascade sentence in `labels.jsonl` carries its kind, so recall can be reported per kind. Besides random KEEP negatives, each positive pair draws hard negatives whose k holds a word of the same kind (a country name, a pronoun, a number) but does not depend on j, so that such a word alone does not predict a cascade. `cascading.views` writes `after.jsonl`, the articles with every direct edit replaced by its gold rewrite, and `evidence.jsonl`, the same with the rewrites added to the evidence.

## Usage

Hidden states of the three files, then the consequence head and the evaluation of each detector:

```bash
D=../datasets/synthetic/split
python -m classifier.extract $D/train.jsonl features/cascade/source
for v in after evidence; do
    python -m classifier.extract $D/$v.jsonl features/cascade/$v --no-tokens
done
python -m cascading.train --features features/cascade/source --features-after features/cascade/after \
    --pairs $D/pairs.jsonl --out heads/consequence

python -m cascading.evaluate --data $D/train.jsonl --labels $D/labels.jsonl --detector mention --out eval/mention.json
python -m cascading.evaluate --data $D/train.jsonl --labels $D/labels.jsonl --detector consequence \
    --features features/cascade/source --features-after features/cascade/after \
    --cascade-head heads/consequence --out eval/consequence.json
python -m cascading.evaluate --data $D/train.jsonl --labels $D/labels.jsonl --detector staleness_delta \
    --features features/cascade/source --features-after features/cascade/after \
    --staleness-head heads/staleness/head.pt --out eval/staleness_delta.json
python -m cascading.evaluate --data $D/train.jsonl --labels $D/labels.jsonl --detector edited_as_evidence \
    --features-evidence features/cascade/evidence --classifier-head heads/classifier/layer32/head.pt \
    --out eval/edited_as_evidence.json
```

The evaluation repairs the direct edits of each article with their gold rewrites, sets Δ to the direct edits, and reports cascade recall for cascades that share a mention with a direct edit (Group A) and for those that do not (Group B), with precision and flag rate.
