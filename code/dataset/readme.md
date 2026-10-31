# Dataset

This directory builds wiki-wrong, the synthetic benchmark that trains and
scores DiffusEdit, and keeps the remains of the FRUIT work that came before
it. The data files are not in git. The finished benchmark is published at
https://huggingface.co/datasets/gamemaker1/wiki-wrong, and `split/` holds the
same files locally.

## Build order

The benchmark is built in six steps, each a script in this directory, and
the working files of every step go under `regen/`.

1. `generate.py` reads Wikipedia leads from the `wikimedia/wikipedia`
   dataset on the Hugging Face Hub, chooses sentences to edit or drop, swaps
   named entities and numbers with the mutators in `mutators/`, injects
   cascades, and writes evidence snippets of the simple forms in
   `evidence.py`. `schema.py` defines the record, and `validate.py` checks a
   file against it and against the tokenizer of the pipeline.
2. `rebuild.py` removes the surface patterns of the first generated file
   that gave the labels away, such as fixed snippet titles and a short list
   of person names, and draws the swapped values from the corpus instead.
3. `tasks.py` turns the rebuilt records into task files for the evidence
   writers, one file per 25 articles, under `regen/in/`. Each task lists the
   sentences of an article with their labels and the values that changed.
4. The writers are Claude Sonnet agents run from `prompt.md`, the brief,
   inside Claude Code workflows. They write the passages of each task file
   to `regen/out/`.
5. `merge.py` joins the written passages to the records and keeps an
   article only if its passages pass every check in its docstring. Rejected
   articles go back to the writers with the reason.
6. `reverse.py` makes the real Wikipedia sentence the outdated source and
   the mutated sentence the gold update, and rewrites the evidence to state
   the mutated values. `split.py` then assigns every record to train,
   validation or test by a hash of its seed article.

`probe.py` fits linear classifiers over surface features of a file to show
how much of the labels those features predict, which is how the shortcuts
of step 2 were found.

## Files

```
split/       train.jsonl, val.jsonl, test.jsonl, with views/ and links/ made by the pipeline
regen/       the working files of the build, from the first generated file to the reversed records
fruit/       the FRUIT loader, the collation script and the label audit
collated/    the collated FRUIT splits that the audit reviewed
archive/     earlier FRUIT subsets and the raw files
```

`fruit/load.py` reads the FRUIT TFRecord files without TensorFlow, and
`fruit/collate.py` builds the collated splits from them. `fruit/audit/`
holds the sheets and the verdicts of the hand audit of the collated labels,
with its own readme.
