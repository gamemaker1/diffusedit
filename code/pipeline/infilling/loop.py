"""End-to-end DiffusEdit inference, Algorithm 1.

One frozen pass over E + A gives the hidden states H. The sentence classifier
labels every sentence, a DROP whose probability clears the gate tau_d is
deleted, and a DROP that does not clear it becomes EDIT. Then, for up to
--rounds rounds:

    1. Every sentence of the EDIT set is masked by the staleness head through
       the masking policy of Section 7.1, skipping tokens that have used up
       their remask budget C_max, and repaired left to right by
       infilling.repair. Delta_r is the set of repaired and deleted sentences.
    2. The article is encoded again (H after the round).
    3. The cascade detector flags sentences outside Delta_r. Staleness delta
       compares the masking head's token scores before and after the round.
       The classifier,
       under the new states, deletes a flagged sentence whose DROP
       probability clears tau_d. The other flagged sentences form the EDIT
       set of the next round.
    4. The loop ends when no sentence is flagged or no flagged sentence has a
       maskable token left.

With --second-pass, a round that repairs two or more sentences repairs each
of them again, from its tokens before the round and at the same spans, with
the round's other repairs placed before the evidence. A cascade sentence has
no snippet of its own, so its fix can only come from the direct edit, and the
first pass of a sentence k before its direct edit j in the article never
reads the repaired j. The added snippets are dropped after the round.

With --detector edited_as_evidence, the sentences repaired in round r are
appended to the evidence before the encoding of step 2, so round r + 1 reads
the k + p snippets. With --detector none the loop is one round.

Oracle runs replace one stage with the gold answer. --labels oracle takes the
sentence labels from the data, also for flagged sentences, so a flagged gold
KEEP sentence is left alone. --masks oracle masks the derived stale tokens of
a sentence that no round has changed yet, and nothing in a sentence whose
gold label is not EDIT. A changed sentence falls back to the staleness head
when one is given.

Output directory as infilling.run, and each line of results.jsonl also holds
"rounds", one entry per round with the sentences repaired, deleted and
flagged, which infilling.metrics --cascades reads.

Usage:
    python -m infilling.loop ../datasets/synthetic/split/train.jsonl runs/loop \\
        --classifier-head heads/classifier/layer32/head.pt \\
        --staleness-head heads/staleness_attention/head.pt \\
        --detector consequence --cascade-head heads/consequence \\
        --ids heads/staleness/heldout_ids.json --limit 50
"""

import argparse
import itertools
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from classifier.extract import Stop, attach, select
from classifier.inputs import DROP, EDIT, KEEP, LABELS, SEPARATOR, build, encode, evidence_ids, read_jsonl
from classifier.train import decide
from infilling.model import Model
from infilling.ner import path_of as entity_path
from infilling.repair import repair_sentence
from infilling.run import arguments, check, done_ids, load_head, settings_of
from infilling.score import Verifier
from infilling.spans import Article, evidence_words, oracle_spans, policy_spans, screen, snap, word_starts

DETECTORS = ("none", "mention", "edited_as_evidence", "staleness_delta", "consequence")


class Classifier:
    """The sentence head of classifier.train over pooled sentence states."""

    def __init__(self, path, device, tau_d=None):
        ckpt = torch.load(path, map_location=device)
        weight = ckpt["state_dict"]["weight"]
        self.head = torch.nn.Linear(weight.shape[1], weight.shape[0]).to(device)
        self.head.load_state_dict(ckpt["state_dict"])
        self.head.eval()
        self.mean, self.std = ckpt["mean"].to(device), ckpt["std"].to(device)
        self.layer = ckpt["layer"]
        self.tau_d = ckpt["tau_d"] if tau_d is None else tau_d

    @torch.inference_mode()
    def __call__(self, pooled):
        """(probabilities [n, 3], labels [n]) with the DROP gate applied."""
        probs = self.head((pooled - self.mean) / self.std).softmax(-1)
        return probs.cpu().numpy(), decide(probs, self.tau_d).cpu().numpy()


def encode_layers(model, article, layers):
    """Hidden states [L, d] of the article's current sequence at every layer in `layers`."""
    ids = torch.from_numpy(article.sequence().astype(np.int64))[None].to(model.device)
    store = {}
    handles = attach(model.blocks, sorted(layers), store)
    try:
        with torch.inference_mode():
            model.net(input_ids=ids)
    except Stop:
        pass
    finally:
        for handle in handles:
            handle.remove()
    model.passes["encode"] += 1
    return {layer: store[layer][0].float() for layer in layers}


def sentence_spans(article):
    spans, start = [], len(article.prefix)
    for sentence in article.sentences:
        spans.append((start, start + len(sentence)))
        start += len(sentence)
    return spans


def pool(hidden, spans):
    """Mean state of each sentence, zeros for a deleted one."""
    return torch.stack([hidden[a:b].mean(0) if b > a else hidden.new_zeros(hidden.shape[1]) for a, b in spans])


def set_evidence(article, tok, evidence):
    """Rebuild the prefix [BOS] + evidence + separator. Returns the evidence token range."""
    ids, _ = evidence_ids(tok, evidence)
    bos = list(article.bos)
    article.prefix = np.asarray(bos + ids + encode(tok, SEPARATOR), dtype=np.int64)
    return (len(bos), len(bos) + len(ids)) if ids else None


def build_detector(args, device):
    """The cascade detector and the layer of the pooled states it reads.

    Staleness delta re-scores tokens with the masking head (Eq. 7), so it
    reads token scores, not pooled states.
    """
    if args.detector == "none":
        return None, None
    if args.detector == "staleness_delta":
        from cascading.detectors import StalenessDeltaDetector
        return StalenessDeltaDetector(tau2=args.tau2), None
    from cascading.evaluate import build
    ns = SimpleNamespace(detector=args.detector, staleness_head=None, tau2=args.tau2,
                         cascade_head=args.cascade_head, tau_c=args.tau_c, classifier_head=args.classifier_head)
    detector, layer, _ = build(ns, device)
    return detector, layer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arguments(parser)
    parser.set_defaults(masks="predicted")
    parser.add_argument("--labels", choices=["predicted", "oracle"], default="predicted",
                        help="sentence labels from the classifier or from the data")
    parser.add_argument("--classifier-head", help="head.pt of classifier.train, for --labels predicted")
    parser.add_argument("--tau-d", type=float, help="DROP gate; default the head's tuned tau_d")
    parser.add_argument("--rounds", type=int, default=3, help="round cap R")
    parser.add_argument("--detector", choices=DETECTORS, default="staleness_delta")
    parser.add_argument("--cascade-head", help="directory of cascading.train, for --detector consequence")
    parser.add_argument("--tau2", type=float, default=0.5, help="staleness delta threshold")
    parser.add_argument("--tau-c", type=float, help="consequence threshold; default the head's tuned tau_c")
    parser.add_argument("--second-pass", action="store_true",
                        help="repair each sentence of a round again with the round's other repairs as evidence")
    args = parser.parse_args()
    check(parser, args)
    needs = {"consequence": "cascade_head", "staleness_delta": "staleness_head", "edited_as_evidence": "classifier_head"}
    if args.detector in needs and not getattr(args, needs[args.detector]):
        parser.error(f"--detector {args.detector} needs --{needs[args.detector].replace('_', '-')}")
    if args.labels == "predicted" and not args.classifier_head:
        parser.error("--labels predicted needs --classifier-head")
    if args.detector == "none":
        args.rounds = 1

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "data": str(Path(args.data).resolve())}
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise SystemExit(f"{out} holds a run with a different configuration")
    else:
        config_path.write_text(json.dumps(config, indent=2))
    results_path = out / "results.jsonl"
    finished = done_ids(results_path)

    settings = settings_of(args)
    entities = {}
    if args.evidence_copy and entity_path(args.data).exists():
        entities = json.loads(entity_path(args.data).read_text())
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = Model(args.model, args.quant, tok)
    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    verifier = Verifier(args.verifier, model.device, args.support_index) \
        if args.evidence and args.scorer != "pll" else None
    head, stale_layer, tau = None, None, args.tau
    if args.staleness_head:
        head, stale_layer, tuned_tau = load_head(args.staleness_head, model.device)
        tau = tuned_tau if args.tau is None else args.tau
    classifier = Classifier(args.classifier_head, model.device, args.tau_d) if args.classifier_head else None
    detector, detector_layer = build_detector(args, model.device)
    from cascading.detectors import EditedAsEvidenceDetector
    carry = EditedAsEvidenceDetector()
    layers = {layer for layer in (stale_layer, detector_layer, classifier and classifier.layer) if layer}
    print(f"labels {args.labels}, masks {args.masks}, detector {args.detector}, rounds {args.rounds}, "
          f"layers {sorted(layers)}, tau {tau}, tau_d {classifier and classifier.tau_d}")

    if args.ids:
        wanted = set(json.loads(Path(args.ids).read_text()))
        chosen = (r for r in read_jsonl(args.data) if r["id"] in wanted)
    else:
        chosen = select(args.data, args.sample, args.seed)
    records = list(itertools.islice(chosen, args.limit))
    todo_records = [r for r in records if r["id"] not in finished]
    print(f"{len(records)} articles, {len(finished)} already done")

    totals, t0 = Counter(), time.time()
    with open(results_path, "a") as results:
        for n_done, record in enumerate(todo_records, 1):
            start, stats = time.time(), Counter()
            model.passes.clear()
            source = record if args.evidence else {**record, "evidence": []}
            example = build(source, tok)
            article = Article.from_example(example, tok.bos_token_id)
            original = [s.copy() for s in article.sentences]
            gold = list(example.labels)
            evidence = list(source["evidence"])
            evidence_text = [f"{s['title']}: {s['text']}" for s in evidence]
            ev_range = (example.evidence_spans[0][0], example.evidence_spans[-1][1]) if example.evidence_spans else None
            n = len(article.sentences)

            def text(j):
                return tok.decode(article.sentences[j].tolist()).strip()

            def token_scores(states, spans, ev_range):
                """The staleness head's score of every token, one array per sentence, or None."""
                if head is None:
                    return None
                view = SimpleNamespace(spans=spans, evidence_spans=[ev_range] if ev_range else [])
                with torch.inference_mode():
                    pi = head(states[stale_layer], view).cpu().numpy()
                return [pi[a:b] for a, b in spans]

            def classify(states):
                if args.labels == "oracle":
                    return None, np.asarray(gold)
                return classifier(pool(states[classifier.layer], sentence_spans(article)))

            states = encode_layers(model, article, layers) if layers else {}
            probs, labels = classify(states)
            initial = [LABELS[int(y)] for y in labels]
            deleted = [j for j in range(n) if labels[j] == DROP]
            for j in deleted:
                article.drop(j)
            todo = [j for j in range(n) if labels[j] == EDIT]
            dirty = bool(deleted)
            rounds, edits = [], []

            for r in range(1, args.rounds + 1):
                if not todo:
                    break
                if dirty and layers:
                    states = encode_layers(model, article, layers)
                dirty = False
                spans_now = sentence_spans(article)
                scores = token_scores(states, spans_now, ev_range)
                previous = [text(j) for j in range(n)]
                before = {layer: pool(states[layer], spans_now) for layer in layers}
                repaired, saved = [], {}
                for j in todo:
                    counts = article.counts[j]
                    if args.masks == "oracle" and np.array_equal(article.sentences[j], original[j]):
                        stale = example.stale[j] if example.stale[j] is not None else np.zeros(len(counts), dtype=bool)
                        spans = oracle_spans(stale, counts, args.c_max)
                    elif scores is not None:
                        spans = policy_spans(scores[j], counts, tau, args.c_max, stats)
                    else:
                        stats["no_oracle_mask"] += 1
                        continue
                    if args.whole_words and spans:
                        pieces = [tok.decode([int(t)]) for t in article.sentences[j]]
                        spans = snap(spans, word_starts(pieces), stats)
                    if args.masks != "oracle" and spans:
                        texts = [tok.decode(article.sentences[j][a:b].tolist()) for a, b in spans]
                        spans = screen(spans, texts, evidence_words(evidence_text), args.evidence_veto,
                                       args.max_spans, stats)
                    if not spans:
                        stats["no_mask"] += 1
                        continue
                    stats["sentences"] += 1
                    masked = [tok.decode(article.sentences[j][a:b].tolist()) for a, b in spans]
                    saved[j] = (article.sentences[j], article.counts[j], spans)
                    decisions = repair_sentence(model, verifier, article, j, spans, evidence_text,
                                                tok, settings, generator, stats, entities.get(record["id"]))
                    repaired.append(j)
                    edits.append({"round": r, "sentence": j, "spans": [list(s) for s in spans],
                                  "masked": masked, "decisions": decisions})
                if args.second_pass and len(repaired) > 1:
                    # The other repairs of the round go before the evidence, as in
                    # edited_as_evidence, for the sampler and for the verifier.
                    prefix, round_edits = article.prefix, {e["sentence"]: e for e in edits if e["round"] == r}
                    for j in repaired:
                        extra = carry.augment_evidence(evidence, {i: text(i) for i in repaired if i != j}, r)
                        set_evidence(article, tok, extra)
                        first = text(j)
                        article.sentences[j], article.counts[j], spans = saved[j]
                        decisions = repair_sentence(model, verifier, article, j, spans,
                                                    [f"{s['title']}: {s['text']}" for s in extra],
                                                    tok, settings, generator, Counter(), entities.get(record["id"]))
                        round_edits[j]["first_pass"] = round_edits[j]["decisions"]
                        round_edits[j]["decisions"] = decisions
                        stats["second_pass"] += 1
                        stats["second_pass_changed"] += text(j) != first
                    article.prefix = prefix
                delta = set(repaired) | set(deleted)
                entry = {"round": r, "repaired": repaired, "deleted": deleted, "flagged": []}
                rounds.append(entry)
                if detector is None or r == args.rounds or not delta:
                    break

                if args.detector == "edited_as_evidence" and repaired:
                    evidence = detector.augment_evidence(evidence, {j: text(j) for j in repaired}, r)
                    ev_range = set_evidence(article, tok, evidence)
                    evidence_text = [f"{s['title']}: {s['text']}" for s in evidence]
                states = encode_layers(model, article, layers) if layers else {}
                spans_after = sentence_spans(article)
                current = [text(j) for j in range(n)]
                inputs = {"previous": previous}
                if detector_layer:
                    inputs["pooled_before"] = before[detector_layer].cpu().numpy()
                    inputs["pooled_after"] = pool(states[detector_layer], spans_after).cpu().numpy()
                if args.detector == "staleness_delta":
                    inputs["token_scores_before"] = scores
                    inputs["token_scores_after"] = token_scores(states, spans_after, ev_range)
                probs, labels = classify(states)
                if probs is not None:
                    inputs["classifier_probs"] = probs
                flagged = {k for k in detector.flag(current, delta, **inputs) if len(article.sentences[k])}
                entry["flagged"] = sorted(flagged)
                deleted, todo = [], []
                for k in sorted(flagged):
                    if labels[k] == DROP:
                        article.drop(k)
                        deleted.append(k)
                    elif args.labels == "predicted" or labels[k] == EDIT:
                        todo.append(k)
                dirty = bool(deleted)
                if not any((article.counts[k] < args.c_max).any() for k in todo):
                    if deleted:
                        rounds.append({"round": r + 1, "repaired": [], "deleted": deleted, "flagged": []})
                    break

            stats.update({f"passes_{k}": v for k, v in model.passes.items() if k != "check"})
            stats["rounds"] = len(rounds)
            if verifier is not None:
                stats["verifier_pairs"] = verifier.pairs
                verifier.pairs = 0
            line = {
                "id": record["id"],
                "labels": initial,
                "source": [s.strip() for s in record["source_sentences"]],
                "output": [text(j) for j in range(n)],
                "rounds": rounds,
                "edits": edits,
                "stats": dict(stats),
                "seconds": round(time.time() - start, 2),
            }
            results.write(json.dumps(line) + "\n")
            results.flush()
            totals.update(stats)
            rate = n_done / (time.time() - t0)
            print(f"{len(finished) + n_done}/{len(records)}  {1 / rate:.1f} s/article  "
                  f"eta {(len(todo_records) - n_done) / rate / 3600:.1f} h  rounds {totals['rounds']}  "
                  f"spans {totals['spans']}", flush=True)


if __name__ == "__main__":
    main()
