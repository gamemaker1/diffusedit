# DiffusEdit: preliminary results on the synthetic train set

The FRUIT experiments of `docs/internal/progress/results.md` were repeated on a synthetic train set with exact labels. This note describes how the set was built, the checks it passes, and the scores of the heads and of infilling. The metric definitions are those of the FRUIT note. Every number comes from a 10% hold-out of the synthetic articles. Raw outputs are in `code/pipeline/outputs/jl/synthetic/`.

## 1. Why the first synthetic file was rebuilt

The first file, `train_30k_diverse.jsonl`, has 30,000 articles and 101,412 sentences (54,830 KEEP, 36,982 EDIT, 9,600 DROP). Its surface form gives the labels away in four ways.

| Shortcut | Count |
| --- | --- |
| A DROP sentence is quoted in a snippet titled "Correction Notice", and no other snippet has that title | 9,600 of 9,600 DROP sentences |
| The gold rewrite of an EDIT sentence appears word for word in a snippet, while KEEP sentences have no snippet | 33,755 of 36,982 EDIT sentences |
| The stale span of an EDIT sentence is one of 12 fixed person names, whatever text it replaced ("born in Elena Rostova, Chile") | 76.6% of stale spans |
| An EDIT sentence ends in an appended template sentence ("As a result, the operation was classified as ...") | 280 sentences |

A shortcut probe measures how far such patterns go. The probe is a linear classifier that never reads the text for meaning. It sees a bag of the sentence's words and word pairs, and eight flags that describe string matches between the sentence and the evidence (for example "the sentence appears inside a snippet"). `code/datasets/synthetic/probe.py` trains it on 90% of the articles and reports macro-F1 on the other 10%. On the first file the probe reaches macro-F1 0.934, and DROP F1 0.997 from the flags alone.

## 2. How the train set was built

### 2.1 Corruptions (`rebuild.py`)

`code/datasets/synthetic/rebuild.py` keeps the gold rewrites of the first file and makes every EDIT source sentence again.

- **Entity swaps.** spaCy `en_core_web_sm` 3.8.0 tags the entities of each gold rewrite, with the 13 types of `datasets/collate.py`. One to three entities per sentence (probabilities 0.6, 0.3, 0.1) are replaced by an entity of the same type, the same word count (up to 3) and the same use of digits, drawn from the other gold rewrites. A year moves by 1 to 15 years. An integer becomes another integer of the same length.
- **DROP to EDIT.** A DROP sentence stays DROP with probability 0.3. The others become EDIT: the sentence is the gold rewrite, and its source is a corruption of it.
- **No entity.** A gold rewrite without an entity becomes KEEP.
- **Suffix.** The 280 appended template sentences are removed.

The result, `train_30k_v2.jsonl`, has 58,846 KEEP, 39,717 EDIT and 2,849 DROP sentences, and 25,929 distinct stale strings. The 12 most frequent cover 5.8% of stale spans.

### 2.2 Evidence passages (`writer_brief.md`, `merge_evidence.py`)

Language model agents wrote the evidence. Each passage reads as an excerpt of another document, such as an encyclopedia article, a news clipping or a press release, and carries the title of that document. `code/datasets/synthetic/writer_brief.md` holds the full instructions. In short:

- For an EDIT sentence, a passage states the correct fact and contains the gold text of every stale span. No passage contains the stale text.
- For a DROP sentence, a passage gives a concrete reason why the sentence no longer holds.
- About half of the KEEP sentences get a consistent passage, so a related passage does not signal a change.
- About one article in three gets a background passage that bears on no sentence.
- No passage copies seven or more consecutive words of a sentence.

Two pilots on the same 100 articles chose the writer model.

| Writer | Articles passing all checks | Passages copying seven or more words |
| --- | --- | --- |
| Claude Haiku 4.5 | 8 of 100 | 204 of 346 |
| Claude Sonnet 5.5 | 93 of 100 | 8 of 259 |

100 Sonnet agents then wrote passages for the first 10,000 articles of `train_30k_v2.jsonl`, 100 articles per agent. `merge_evidence.py` keeps an article only if its passages pass every check.

| Outcome | Articles |
| --- | --- |
| Kept | 9,183 |
| Rejected, a passage copies seven or more words | 745 |
| Rejected, a gold span is absent from the passages about its sentence | 70 |
| Rejected, a passage contains stale text | 2 |

### 2.3 The train set

`train_written.jsonl` has 9,183 articles and 31,315 sentences (18,628 KEEP, 11,813 EDIT, 874 DROP). It has 19,991 passages, 2.2 per article and 24 words each on average, under 19,972 distinct titles. 12.0% of EDIT-sentence characters are stale. `examples.md` shows five articles.

### 2.4 Shortcut probe

| Probe features | First file | Rebuilt, evidence copies the gold sentence | Rebuilt, written passages |
| --- | --- | --- | --- |
| Sentence words only | 0.590 | 0.399 | 0.375 |
| Evidence-match flags only | 0.793 | 0.839 | 0.399 |
| Both | 0.934 | 0.839 | 0.418 |

Each cell is macro-F1 on the probe's hold-out. The middle column is `train_30k_v2.jsonl` with the placeholder evidence that `rebuild.py` writes. On the train set the probe's DROP F1 is 0.034 at best.

## 3. Setup of the runs

The runs use the scripts, model revision and settings of the FRUIT runs, on the same A100 40 GB instance. `jobs/jl/run_synthetic.sh` runs the stages of `run_all.sh`, `run_fixes.sh` and `run_heads.sh` on `train_written.jsonl` and writes to a separate directory. All stages finished in 34 minutes with no failure.

- **Hold-out.** `split_by_article` with seed 42 holds out 10% of the articles: 3,116 sentences (1,838 KEEP, 1,194 EDIT, 84 DROP).
- **Extraction.** 9,183 articles in 5 min 43 s (31.5 articles per second), no non-finite values. The features hold 31,315 sentence rows, 19,991 passage rows and 347,903 EDIT-sentence tokens.
- **Infilling articles.** 659 of the 678 articles held out by the staleness head contain a supported EDIT sentence. The runs use the first 50.

Two evaluation scripts read fields that only FRUIT records have. `infilling/entities.py` and `infilling/significance.py` now take the gold rewrite through `infilling.evaluate.gold`, which reads either schema.

## 4. Sentence classifier

| Model | Macro-F1 | KEEP F1 | EDIT F1 | DROP F1 | DROP precision | Macro-F1 on FRUIT |
| --- | --- | --- | --- | --- | --- | --- |
| MiniLM baseline | 0.443 | 0.563 | 0.564 | 0.201 | 0.163 | 0.433 |
| LLaDA layer 24 | **0.870** | 0.946 | 0.921 | 0.742 | 0.787 | 0.499 |
| LLaDA layer 32 | 0.708 | 0.871 | 0.833 | 0.419 | 0.422 | 0.490 |

| Evidence input | Layer 24 | Layer 32 |
| --- | --- | --- |
| None | **0.870** | 0.708 |
| Mean | 0.863 | **0.751** |
| Nearest | 0.854 | 0.745 |
| Attention | 0.864 | 0.742 |

Both tables use seed 42. Over five seeds (42, 1, 2, 3, 4), each with its own hold-out:

| Model | Macro-F1, mean ± standard deviation |
| --- | --- |
| MiniLM baseline | 0.443 ± 0.018 |
| LLaDA layer 24, cross-entropy | 0.849 ± 0.015 |
| LLaDA layer 24, focal loss | 0.839 ± 0.012 |
| LLaDA layer 32, cross-entropy | 0.691 ± 0.010 |
| LLaDA layer 32, focal loss | 0.687 ± 0.002 |

## 5. Token staleness head

The held-out part has 14.9% stale tokens.

| Metric | Synthetic | FRUIT |
| --- | --- | --- |
| AUROC | 0.993 | 0.645 |
| PR-AUC | 0.972 | 0.405 |
| τ tuned for F1 | 0.85 | 0.40 |
| Precision at the tuned τ | 0.935 | 0.344 |
| Recall at the tuned τ | 0.898 | 0.789 |
| `tau_match` | 0.80 | 0.60 |
| Precision at `tau_match` | 0.915 | 0.415 |
| Recall at `tau_match` | 0.914 | 0.377 |

Over five seeds, AUROC is 0.992 ± 0.001 with binary cross-entropy and 0.990 ± 0.001 with focal loss. Token F1 is 0.910 ± 0.005 and 0.903 ± 0.004.

## 6. Infilling

The three runs share 50 held-out supported articles and 99 repaired sentences, 95 of them with new gold words. Sentence labels are the derived ones, as on FRUIT. The unchanged source sentence (copy) scores ROUGE-L 0.867.

| Run | ROUGE-L | Pool ROUGE-L | Hit | Coverage, chosen | Masked share of characters | Time per article |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle masks | **0.940** | 0.976 | 80.0% | 79.6% | 14.4% | 13.0 s |
| Oracle masks, no evidence | 0.905 | 0.953 | 49.5% | 46.9% | 14.4% | 6.3 s |
| Predicted masks, τ = 0.80 | 0.926 | 0.967 | 77.9% | 77.7% | 15.8% | 14.1 s |

The stale share of characters in these sentences is 13.3%. The deletion candidate won none of 164 span decisions in the predicted-mask run.

## 7. Findings

1. With exact labels and evidence that states the change, the layer-24 classifier reaches macro-F1 0.870, against 0.499 on FRUIT. The MiniLM baseline stays at its FRUIT level (0.443 against 0.433), so the gain comes from the LLaDA states of the evidence-conditioned pass.
2. Layer 24 exceeds layer 32 by 0.162 macro-F1. On FRUIT the difference was 0.009.
3. The extra evidence inputs change layer 24 by at most 0.016, within the seed spread of 0.015. At layer 32 they add 0.034 to 0.043.
4. The staleness head separates stale from correct tokens (AUROC 0.993). On FRUIT its AUROC was 0.645.
5. Predicted masks raise ROUGE-L from 0.867 (copy) to 0.926. On FRUIT they lowered it from 0.782 to 0.678.
6. Evidence raises hit from 49.5% to 80.0% and ROUGE-L from 0.905 to 0.940 under oracle masks.

## 8. Limits

- **Label balance.** DROP is 2.8% of sentences. The hold-out has 84 DROP sentences, so DROP F1 0.742 at layer 24 rests on 59 correct decisions.
- **Size.** The train set has 9,183 articles, 11% of the 84,117 FRUIT articles.
- **Edit type.** Every edit is an entity swap, and the corrected value appears in a passage. FRUIT edits are free rewrites, and 16.9% of those with new words are supported.
- **Implausible swaps.** A replacement matches the type that spaCy assigned, and that type is sometimes wrong ("Solaris 11" for a date). The staleness head may partly detect text that is odd in context, without the evidence. A run without evidence in the extraction pass would measure this share. It has not been run.
- **Invented reasons.** The dropped sentences are true Wikipedia sentences, so the passages that contradict them state invented facts.
- **Segmentation.** The first file splits some sentences at abbreviations ("It is led by Victor R." and "Huerta."). The split was left as it is.
- **Transfer.** No model trained here was evaluated on FRUIT.

## 9. Open items

- Entity metrics and paired significance tests for the three infilling runs. Both run without the GPU.
- The staleness score analysis. `analysis/stale_analysis.py` hard-codes the FRUIT feature directory.
- Extraction without evidence, for the control named in Section 8.
- Passages for the other 20,000 articles of `train_30k_v2.jsonl`.
- Evaluation on FRUIT of heads trained on the synthetic set.

## 10. Reproduction

| Step | Script |
| --- | --- |
| Corruptions | `python rebuild.py train_30k_diverse.jsonl train_30k_v2.jsonl` (in `code/datasets/synthetic/`, needs spaCy 3.8 with `en_core_web_sm` 3.8.0) |
| Passages | one writer agent per 100 articles, instructed by `writer_brief.md` |
| Checks and merge | `python merge_evidence.py train_30k_v2.jsonl <passages dir> train_written.jsonl` |
| Shortcut probe | `python probe.py <file>.jsonl` (needs scikit-learn) |
| Upload and start | `jobs/jl/https/deploy_synthetic.sh` |
| All stages on the instance | `jobs/jl/run_synthetic.sh` |
| Wait, pull, verify, pause | `jobs/jl/https/finish_synthetic.sh` |

The dataset files and `jobs/` are ignored by git. The Jupyter server of the instance rejects a single upload of the 22 MB dataset with HTTP 500, so `deploy_synthetic.sh` sends it gzipped in 4 MB parts.
