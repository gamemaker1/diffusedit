# DiffusEdit

Reference articles go out of date as events develop, and updating them by hand
is slow. Regenerating the article with a fine-tuned language model automates
the update, but it trains every parameter of the model and leaves no record of
which tokens changed. DiffusEdit instead edits the article in place. Small
trained networks, called classifiers, read the hidden states of a frozen
masked diffusion language model,
[LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base), and decide
which sentences to keep, edit, or delete, and which tokens inside them are
stale. The frozen model rewrites only the stale positions, a reranker chooses
among the rewrites, and the system repeats the process on sentences that the
rewrites themselves made wrong. Training labels come from wiki-wrong, a
synthetic benchmark of Wikipedia leads with known edits, and evaluation
measures update quality, minimality, and coverage of the cascading edits.

## Links

The data, the trained classifiers, and the fine-tuned baseline are published
on the Hugging Face Hub.

| Repository | Contents |
| --- | --- |
| [gamemaker1/wiki-wrong](https://huggingface.co/datasets/gamemaker1/wiki-wrong) | the benchmark, with its train, validation, and test splits |
| [gamemaker1/diffusedit](https://huggingface.co/gamemaker1/diffusedit) | the sentence, staleness, and cascade classifiers, and the reranker |
| [gamemaker1/t5edit](https://huggingface.co/gamemaker1/t5edit) | Flan-T5-large fine-tuned to rewrite the article, the upper bound |

## Layout

```
code/dataset/    builds wiki-wrong, and keeps the earlier FRUIT work
code/pipeline/   the classifiers, the edit loop, the reranker, and the evaluation
docs/            the proposal, the reports, the slides, and the papers read
```

The readme in each directory under `code/` lists its build or run order and
the command for each step.
