# DiffusEdit: first experiments on the train split

Runs on one A100 40 GB (JarvisLabs instance "DiffusEdit Training"). Every number below comes from articles of the train split. No job read `val.jsonl` or `test.jsonl`. Raw outputs are in `code/pipeline/outputs/jl/`.

## 1. Setup

### 1.1 Data and held-out split

The input is `code/datasets/collated/train.jsonl`: 84,117 articles, 384,788 source sentences (174,873 KEEP, 162,053 EDIT, 47,862 DROP) and 516,999 evidence snippets. The labels are silver. An automatic aligner derived them from Wikipedia revisions, and the team audit estimates 40 to 60% label noise per class.

All heads hold out the same 10% of train articles. `classifier.train.split_by_article` draws them with seed 42, so no article has sentences on both sides. The held-out part has 8,256 articles, 38,267 sentences and 531,706 EDIT-sentence tokens. It drives early stopping, threshold tuning and every reported number.

An EDIT sentence is *supported* when every word of its gold rewrite that is absent from the source sentence appears in the source or in the article's evidence. Only 16.9% of EDIT sentences with new words are supported (57 of 338 in the first infilling run), so most gold rewrites contain facts the input does not hold. 2,280 held-out articles contain at least one supported EDIT sentence, 2,557 such sentences in total. Their ids are in `outputs/jl/supported_ids.json`.

### 1.2 Feature extraction

`classifier.extract` runs LLaDA-8B-Base (revision `0f2787f`) once per article over the sequence [BOS] + evidence + separator + article. The weights are bf16, not quantized. Forward hooks read the output of blocks 24 and 32 and stop the pass after block 32. The script stores three things:

- the mean of the token states of every source sentence, at layers 24 and 32 (the pooled vector of Eq. 4)
- the mean of the token states of every evidence snippet, at layers 24 and 32
- the state of every token of every EDIT sentence at layer 32, with its stale label

The run took 1 h 40 min for 84,117 articles (8 to 14 articles per second) and produced no non-finite values. The features occupy about 55 GB on the instance's `/home` volume.

### 1.3 Code changes made for these runs

| File | Change |
| --- | --- |
| `classifier/extract.py` | bf16 compute on GPUs of compute capability 8.0 or higher, fp16 below |
| `staleness/train.py`, `experiments/baseline_train.py` | Optional `--val`. Without it, the held-out split of Section 1.1. `staleness.train` writes `heldout_ids.json`. |
| `experiments/evidence_train.py` | `--evidence-pool mean, nearest, attention` |
| `staleness/sweep.py` | New. Precision, recall and masked share per τ, and `tau_match` |
| `infilling/run.py` | `--ids` to choose articles. `--whole-words`, on by default. |
| `infilling/spans.py` | `word_starts` and `snap` widen mask spans to word boundaries |
| `infilling/sampler.py` | `sample()` runs under `torch.inference_mode()`. Without it, infilling crashed on the first article. |
| `infilling/evaluate.py` | Gold rewrite from `target_sentences[target_idx]["text"]`. Subsets `supported` and `cited`. `--limit`. |
| `infilling/entities.py` | New. Entity metrics of Section 9.2. |
| `infilling/significance.py` | New. Paired bootstrap and permutation tests. |
| `analysis/stale_analysis.py`, `analysis/stale_scatter.py` | New. Staleness head analysis and plot data. |

## 2. Sentence classifier

### 2.1 Method

The head is one linear layer from the pooled sentence vector to KEEP, EDIT and DROP logits (Eq. 5). Features are standardized with train statistics. The loss is cross-entropy with class weights N / (3 N_c). Training uses AdamW (learning rate 1e-3, weight decay 1e-2, batch 1024) for up to 30 epochs and stops after 3 epochs without a gain in held-out macro-F1. A sentence whose arg-max is DROP is deleted only if P(DROP) ≥ τ_d, and τ_d is tuned on the held-out part over 0.35 to 0.95.

Macro-F1 is the mean of the three per-class F1 scores, so the frequent KEEP class does not dominate it.

The baseline embeds each sentence and each snippet with `sentence-transformers/all-MiniLM-L6-v2` (mean pooling) and trains the same head on the sentence vector concatenated with the mean snippet vector. It reads no LLaDA features.

The evidence ablation appends one evidence vector per sentence to the LLaDA sentence vector:

- **mean**: the mean of the article's snippet vectors
- **nearest**: the article's snippet with the highest cosine similarity to the sentence vector
- **attention**: a learned query from the sentence vector attends over the article's snippets (width 256, at most 32 snippets), and the attention and head train together

### 2.2 Results

The held-out part has 38,267 sentences: 17,652 KEEP, 16,100 EDIT and 4,515 DROP.

| Model | Macro-F1 | KEEP F1 | EDIT F1 | DROP F1 | DROP precision | τ_d |
| --- | --- | --- | --- | --- | --- | --- |
| MiniLM baseline | 0.433 | 0.440 | 0.580 | 0.278 | 0.210 | 0.40 |
| LLaDA layer 24 | **0.499** | 0.558 | 0.626 | 0.315 | 0.240 | 0.45 |
| LLaDA layer 32 | 0.490 | 0.534 | 0.611 | 0.327 | 0.245 | 0.45 |

| Evidence input | Layer 24 | Layer 32 |
| --- | --- | --- |
| None | **0.499** | **0.490** |
| Mean | 0.495 | 0.486 |
| Nearest | 0.486 | 0.474 |
| Attention | 0.488 | 0.481 |

LLaDA layer 24 exceeds the baseline by 0.066 macro-F1. No evidence input raises macro-F1. The attention weights collapse onto one snippet (mean peak weight 0.999). No significance test covers these numbers yet, because the runs saved aggregate metrics only (Section 7).

## 3. Token staleness head

### 3.1 Method

The head is one linear layer with a sigmoid on the layer-32 state of each EDIT-sentence token (Eq. 7). It trains on 4,804,230 tokens, 28.2% stale, with binary cross-entropy and a positive-class weight of 2.54. τ is chosen for token F1 on the held-out part.

Precision is the share of masked tokens that are stale. Recall is the share of stale tokens that get masked. PR-AUC is the area under the precision-recall curve, and AUROC is the probability that a random stale token outscores a random correct one (0.5 for a random scorer).

### 3.2 Results

The held-out part has 531,706 tokens in 16,100 EDIT sentences, and 28.5% of them are stale. PR-AUC is 0.405 and AUROC 0.645 (0.667 averaged within sentences).

| Masking rule | Precision | Recall | Share of tokens masked |
| --- | --- | --- | --- |
| τ = 0.40 (tuned for F1) | 0.344 | 0.789 | 65.3% |
| τ = 0.55 | 0.393 | 0.486 | 35.2% |
| τ = 0.60 (`tau_match`) | 0.415 | 0.377 | 25.8% |
| τ = 0.65 | 0.440 | 0.272 | 17.6% |
| Top 28.5% of each sentence | 0.391 | 0.392 | 28.5% |
| Top k of each sentence, k = true stale count | 0.573 | 0.573 | 28.5% |

`tau_match` is the τ whose masked share is closest to the stale share. Any threshold rule masks the top m tokens of each sentence for some m, so the last row bounds every threshold rule from above. With the true count known for every sentence, 42.7% of masks still land on correct tokens.

The share of stale tokens per score bin rises from 0.060 at scores 0.10 to 0.12 to 0.617 at 0.90 to 0.93. It passes 0.5 only above a score of 0.81, where 2.1% of tokens lie. In the 0.60 bin, 35% of tokens are stale, so the scores are not calibrated probabilities.

Figures are in `outputs/jl/plots/`. `staleness_tokens.png` shows per-token scores in four sentences, `staleness_threshold.png` the score distributions and threshold sweep, and `staleness_scatter.png` the scores against labels, P(stale | score) and a 2-D view of token states.

## 4. Infilling

### 4.1 Method

`infilling.run` performs one repair round (Algorithm 1, lines 2 to 5). DROP sentences are deleted and every EDIT sentence is repaired, both by the derived sentence labels, which is the oracle-sentence-label setting of Section 9.3. The sentence classifier does not feed infilling yet.

1. **Masks.** `oracle` masks the derived stale tokens. `predicted` masks tokens whose staleness score is at least τ, with the fallback, cap and gap rules of Section 7.1.
2. **Whole words.** Each span widens to word boundaries. The first run predates this step.
3. **Candidates.** Spans are decided left to right. Each span is filled at lengths {0, ⌈ℓ/2⌉, ℓ, ℓ+2} (the `full` set in `spans.py`) with T = 8 sampler steps and greedy decoding.
4. **Reranking.** Each candidate scores R = 0.5 · PLL + 0.5 · Ent. PLL is the mean log-probability of each sentence token with that token masked alone. Ent is the highest P(SUPPORTS) of the VitaminC verifier `tals/albert-base-vitaminc-mnli` over the evidence snippets. The deletion candidate wins only by a margin of 0.1.

### 4.2 Metrics

All metrics compare each repaired sentence with its gold rewrite.

- **ROUGE-L** is the F1 of the longest common word subsequence. It is mostly carried by the words that did not change.
- **Hit** is the share of sentences where some candidate contains every new gold word. **Coverage** is the share of new gold words in a text.
- **Entity metrics** (Section 9.2) use the mention extractor of `datasets/collate.py`: spaCy `en_core_web_sm` 3.8.0 entities of 13 types, lowercased, plus every digit string, run on each sentence alone. With G the gold mentions, S the source mentions, O the output mentions and E the evidence mentions:
  - precision is |O ∩ G| / |O|
  - recall is |O ∩ G| / |G|
  - new-entity recall is |O ∩ (G − S)| / |G − S|, the entities the update adds
  - outdated kept is |O ∩ (S − G)| / |S − G|, lower is better
  - fabricated is |O − S − E| / |O|, lower is better

  Each metric sums numerators and denominators over sentences.
- **Copy** is the unchanged source sentence. **Pool** is the candidate with the highest new-entity recall, chosen with the gold rewrite, so it is an upper bound on the reranker.

### 4.3 Runs

| Run | Articles | Masks | Evidence | Whole words | Time per article |
| --- | --- | --- | --- | --- | --- |
| `infill-full-predicted` | first 200 held-out | predicted, τ = 0.40 | yes | no | 80.8 s |
| `fix-oracle` | first 50 of `supported_ids.json` | oracle | yes | yes | 49.1 s |
| `fix-oracle-noev` | same 50 | oracle | no | yes | 13.4 s |
| `fix-predicted` | same 50 | predicted, τ = 0.60 | yes | yes | 90.9 s |

The three `fix-*` runs share 50 articles and 106 repaired sentences, 62 of them supported. In an article of `fix-oracle`, PLL took 1,892 forward passes against 410 for the sampler.

### 4.4 Results on the 50 shared articles

| System | ROUGE-L | Entity precision | Entity recall | New-entity recall | Outdated kept | Fabricated |
| --- | --- | --- | --- | --- | --- | --- |
| Copy | 0.782 | 0.727 | 0.740 | 0.000 | 1.000 | 0.000 |
| Oracle masks | **0.801** | **0.753** | **0.777** | **0.218** | 0.236 | **0.144** |
| Oracle masks, no evidence | 0.779 | 0.701 | 0.716 | 0.084 | 0.197 | 0.229 |
| Predicted masks, τ = 0.60 | 0.678 | 0.578 | 0.587 | 0.134 | 0.472 | 0.247 |
| Pool, oracle masks | n/a | 0.930 | 0.810 | 0.328 | 0.055 | 0.085 |

On the 62 supported sentences, oracle masks reach new-entity recall 0.277 with evidence and 0.031 without it.

| Run | Hit | Coverage, chosen | Masked share of characters | Stale share of characters |
| --- | --- | --- | --- | --- |
| `fix-oracle` | 33.3% | 32.9% | 24.4% | 22.8% |
| `fix-oracle-noev` | 21.6% | 18.9% | 24.4% | 22.8% |
| `fix-predicted` | n/a | n/a | 33.3% | 22.8% |
| `infill-full-predicted` (200 other articles) | 12.4% | 17.4% | 55.3% | 26.6% |

`infill-full-predicted` scored ROUGE-L 0.565 against 0.731 for copy on its 200 articles. Its masks covered twice the stale share and split words: masking " (including Chinese h" left "anzi" behind and produced "Chinese charactersanzi".

### 4.5 Offline reranking

`results.jsonl` stores PLL and Ent for every candidate, so other scoring rules can re-pick among stored candidates without the GPU. Over 1,550 decisions of `infill-full-predicted`, nine rules gave mean ROUGE-L between 0.559 and 0.570, against 0.569 for the current rule. The rules were raw, exponentiated and z-scored PLL with λ from 0.1 to 0.5, PLL alone, Ent alone and a lexical evidence-overlap score. Ent varies little between candidates of one decision (median range 0.002, against 0.374 for PLL). The chosen candidate equals the highest-PLL candidate in 88.5% of decisions. By ROUGE-L, picking with the gold rewrite gains 0.03. By entity metrics the gain is larger: new-entity recall rises from 0.218 to 0.328 and outdated kept falls from 0.236 to 0.055 (Section 4.4).

## 5. Significance tests

`infilling.significance` compares two systems on the same sentences. The article is the unit of resampling, because sentences of one article are not independent.

- **95% CI**: paired bootstrap over the 50 articles, 10,000 resamples, of the difference of the ratio-of-sums metric
- **p**: paired permutation test, with A and B swapped within each article at random, 10,000 permutations, two-sided
- **p_holm**: Holm correction over the 30 tests of one table

Comparisons with copy on new-entity recall, outdated kept and fabricated are always significant, because copy changes nothing and so scores exactly 0, 1 and 0 on them. The other comparisons follow. A * marks p_holm < 0.05.

| Comparison | Metric | A − B | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- | --- |
| Oracle vs copy | ROUGE-L | +0.019 | [+0.005, +0.034] | 0.010 | 0.093 |
| Oracle vs copy | Entity precision | +0.025 | [−0.019, +0.069] | 0.306 | 1.000 |
| Oracle vs copy | Entity recall | +0.037 | [+0.006, +0.077] | 0.052 | 0.417 |
| Oracle vs no evidence | Entity recall | +0.061 | [+0.025, +0.114] | 0.0003 | 0.005 * |
| Oracle vs no evidence | New-entity recall | +0.134 | [+0.038, +0.264] | 0.006 | 0.068 |
| Oracle vs no evidence | Fabricated | −0.085 | [−0.121, −0.050] | 0.0001 | 0.003 * |
| Predicted vs copy | ROUGE-L | −0.105 | [−0.130, −0.078] | 0.0001 | 0.003 * |
| Predicted vs copy | Entity precision | −0.149 | [−0.207, −0.091] | 0.0001 | 0.003 * |
| Predicted vs copy | Entity recall | −0.153 | [−0.215, −0.089] | 0.0002 | 0.003 * |
| Oracle vs predicted | Entity recall | +0.190 | [+0.121, +0.255] | 0.0001 | 0.003 * |
| Oracle vs predicted | New-entity recall | +0.084 | [−0.050, +0.223] | 0.352 | 1.000 |
| Oracle vs predicted | Outdated kept | −0.236 | [−0.356, −0.122] | 0.002 | 0.021 * |
| No evidence vs copy | ROUGE-L | −0.004 | [−0.019, +0.013] | 0.717 | 1.000 |

On the 62 supported sentences, oracle masks with evidence beat oracle masks without it on ROUGE-L by +0.040 (p_holm 0.003) and on entity recall by +0.071 (p_holm 0.019). The full tables are in `outputs/jl/runs/significance.json`.

## 6. Findings

1. Predicted masks make the output worse than the unchanged sentence on ROUGE-L, entity precision and entity recall, with p_holm ≤ 0.003 on all three.
2. The staleness head limits the system. Its AUROC is 0.645, and its precision stays at or below 0.573 under any threshold rule.
3. With oracle masks, the output exceeds copy by 0.019 ROUGE-L and 0.037 entity recall. Neither gain survives the Holm correction over 50 articles.
4. Evidence lowers fabricated entities from 22.9% to 14.4% and raises entity recall by 0.061, both significant. The new-entity recall gain (+0.134) has p_holm 0.068.
5. The candidate pool holds better outputs than the reranker picks: new-entity recall 0.328 against 0.218, and 5.5% of outdated entities kept against 23.6%.
6. No evidence input improves the sentence classifier.

## 7. Open items

- **Classifier and staleness significance tests.** These need per-sentence and per-token predictions exported from the instance, about 10 to 15 minutes of instance time.
- **Statistical power.** With 50 articles, the oracle-vs-copy interval on ROUGE-L is about ±0.015. 200 articles would roughly halve it, at about 2.7 hours of GPU time per oracle run (200 articles at the measured 49.1 s per article).
- **Staleness head.** Per-token evidence features, an MLP over the token state concatenated with the sentence vector and the nearest snippet vector, and training on supported edits only. All three can use the features already on the instance.
- **Reranker.** An entity-aware term, tested offline on the stored candidates first.
- **Unbuilt blocks.** The cascade detectors (Section 7.3), the consequence head and pair mining (Sections 6.4 and 8.5), the multi-round loop (Section 7.4), the classifier-to-infilling wiring, and gold-set evaluation.

## 8. Reproduction

The instance is reachable over HTTPS through its Jupyter URL (`jl list --json`, field `url`). SSH on port 22 is blocked on the IIIT LAN. The instance ID and IP change on every resume. `jobs/jl/https/jup.py` wraps the Jupyter contents API for files and a kernel websocket for shell commands, and needs a Python with `websocket-client`. Uploads must omit the `chunk` field for single-chunk files. A lone `chunk: -1` appends to the existing file.

| Step | Script (in `code/pipeline/`) |
| --- | --- |
| Environment and model download | `jobs/jl/setup.sh` |
| Extraction, classifier, evidence ablation, baseline, staleness, first infilling run | `jobs/jl/run_all.sh` |
| τ sweep, pooling variants, `fix-*` runs | `jobs/jl/run_fixes.sh` |
| Upload and start, wait and pull and pause | `jobs/jl/https/deploy_fixes.sh`, `jobs/jl/https/finish_fixes.sh`, `jobs/jl/https/pull.py` |
| Entity metrics | `python -m infilling.entities runs/<run> <train.jsonl>` |
| Significance tests | `python -m infilling.significance <train.jsonl> oracle=… noev=… predicted=… --pairs …` |
| Staleness analysis and plots | `analysis/stale_analysis.py`, `analysis/stale_scatter.py` (run on the instance) |

`jobs/` is ignored by git through `jobs/.gitignore`. `jobs/jl/https/finish.sh` hard-codes the old instance ID 522760. `finish_fixes.sh` reads the current ID from `jl list`.
