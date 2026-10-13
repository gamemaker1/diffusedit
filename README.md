# Label audit

Hand review of the KEEP/EDIT/DROP labels in the collated train split. The
labels are derived automatically from FRUIT revision pairs, so some are wrong.
This folder measures how many.

## Sheets

| File | Rows | What it holds |
|---|---|---|
| `train_audit.tsv` | 436 | Every sentence of 100 random train articles (seed 42) from the 20,000 extracted. Gives an unbiased error rate. |
| `train_audit_evidence.tsv` | 638 | The evidence snippets of those 100 articles. |
| `flagged.tsv` | 300 | The train sentences whose label a cross-validated classifier disputes most confidently (confident learning, `classifier/noise.py`). Most are the model's mistakes, not label errors. Its `given` labels predate the KEEP fix below. |
| `flagged_evidence.tsv` | | The evidence snippets of the flagged articles. |
| `noise.json` | | The confident-learning estimate these flags came from. |

Blank sheets stay blank. Each reviewer's verdicts go to
`annotations/<name>/<sheet>`, created on first use.

## Reviewing

Python 3.6 or newer, standard library only. Run from this folder:

```
python3 review.py train_audit.tsv --annotator <name> --evidence train_audit_evidence.tsv
python3 review.py flagged.tsv --annotator <name> --evidence flagged_evidence.tsv
```

Each row shows the source sentence with its stale words in red, the aligned
target sentence with new words in green, the neighbouring sentences, and the
evidence. Type a key, then Enter:

| Key | Verdict |
|---|---|
| Enter alone | the given label is right |
| `k` `e` `d` | the right label is KEEP, EDIT or DROP |
| `s` | not a real sentence (a bad split such as `.` or `(image)`) |
| `u` | unsure |
| `n` | add a note |
| `v` | show all evidence, not only the snippets the target cites |
| `b` / `j N` | back one row / jump to row N |
| `q` | save and quit |

On `train_audit.tsv`, an EDIT or DROP verdict opens a second prompt: does the
evidence support the change (`y`, `n`, Enter to skip)?

Every answer is saved at once, and the next run resumes at the first row
without a verdict.

## What the labels mean

- **KEEP**: the sentence survives the revision unchanged. Since the fix of
  28 September 2026, a sentence whose revision changed a name or number is
  EDIT even when the rest is identical.
- **EDIT**: the revision rewrote part of the sentence.
- **DROP**: the revision deleted the sentence.

Judge whether the label describes what the revision did. Record separately,
in the evidence prompt or a note, whether the evidence explains it.

## Results

```
python3 review.py train_audit.tsv --annotator <name> --summary
python3 review.py train_audit.tsv --agreement
```

`--summary` prints the share of labels judged wrong, broken down by given
label. `--agreement` compares every reviewer of a sheet on the rows they share:
raw agreement and Cohen's kappa (agreement corrected for chance).
