# Pipeline

The code of DiffusEdit and of its evaluation. It is a `uv` project. `uv sync`
creates the environment, and every command below runs from this directory as
`uv run python -m <module>`. The data is expected under `../dataset/split`,
as published at https://huggingface.co/datasets/gamemaker1/wiki-wrong, and
the trained classifiers are published at
https://huggingface.co/gamemaker1/diffusedit. Each module starts with a
docstring that states what it computes, the files it reads and writes, and
the command that runs it.

## Flow

One run of the system goes through the packages in this order.

1. `classifier.extract` runs the frozen LLaDA-8B-Base once over every article
   with its evidence and caches the hidden states: one pooled vector per
   sentence at blocks 24 and 32, one vector per token of the edited sentences
   and per evidence token at block 32. `classifier.inputs` builds the input
   sequence and the sentence spans that the extractor and the infilling step
   share.
2. `classifier.train` fits the sentence classifier on the cached sentence
   vectors. `staleness.train` fits the linear token classifier and
   `staleness.attention` the attention token classifier on the cached token
   and evidence vectors. `staleness.sweep` tunes the mask threshold.
3. `cascading.links` derives the cascade pairs and labels from the records,
   `cascading.views` writes the oracle-repaired views whose states the cascade
   classifier reads, and `cascading.train` fits it. `cascading.detectors`
   holds the three detectors the loop can use, and `cascading.evaluate`
   scores a detector on its own.
4. `infilling.run` repairs the edited sentences of a split in one round:
   `infilling.spans` turns token scores into masks and candidate lengths,
   `infilling.sampler` and `infilling.model` fill the masks with the frozen
   model, `infilling.repair` builds the candidate pool of a sentence with
   the infilling rules, and `infilling.score` ranks it. `infilling.ner`
   writes the named entities that the evidence-copy candidates draw from.
   `infilling.loop` runs the rounds with a cascade detector.
5. `infilling.rerank` fits the learned ranking on the logged pools of a
   validation run and applies it to a test run without another model pass.
6. `infilling.evaluate` scores a run at the sentence level,
   `infilling.metrics` at the article level with cascade recall, and
   `infilling.significance` and `infilling.update_rouge` run paired tests
   between runs. `infilling.entities` and `infilling.faithfulness` hold the
   entity and AlignScore metrics of the earlier reports.

## Experiments and analysis

`experiments.editor` fine-tunes the Flan-T5 editor and writes its outputs in
the same per-sentence format as `infilling.run`, so every scorer reads both.
`experiments.baseline_train` fits the sentence-transformer baseline of the
sentence classifier, `experiments.evidence_train` and
`experiments.evidence_extract` the classifier variants with an evidence
summary, and `experiments.stress` builds the stress sets.

Under `analysis`, `values` computes the value-level sentence scores,
`extra` scores editor and infilling runs on the same gold sentences,
`rounds` rebuilds the article after each loop round, `focus` measures where
the attention classifier's weight falls in the evidence, `classifier_tests`
and `staletests` run paired tests between classifiers, and `paper` draws
the figures of the final write-up. The other modules in `analysis` draw the
figures of the earlier reports.
