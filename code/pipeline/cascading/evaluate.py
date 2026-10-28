"""Cascade recall of one detector after an oracle repair of the direct edits (Section 9.2).

The evaluation simulates one round of Algorithm 1. The round repairs the
direct edits of each article with their gold rewrites and nothing else, so Δ
is the direct set and every cascade sentence is still unrepaired. The detector then
flags sentences outside Δ, and cascade recall is the fraction of cascade
sentences it flags, reported for cascades that share a mention with a direct
edit (Group A) and for those that do not (Group B). Mention overlap flags
only sentences that share a mention string, so its Group B recall is 0.

The detectors read these inputs.

    mention             the sentences after the round, and their versions
                        before it
    staleness_delta     the staleness head over --features (before) and
                        --features-after (the after view of cascading.views)
    consequence         the consequence head with u_j from --features-after
                        and u_k from --features
    edited_as_evidence  the classifier head over --features-evidence (the
                        evidence view of cascading.views)

Labels file (JSONL, one record per article, from cascading.links or by hand):
    {
        "article_id": str,
        "direct": [int, ...],       // when absent, cascading.links.direct
        "cascade_sentences": [
            {"sentence_idx": int, "label": "cascade"|"stylistic",
             "shares_mention_with_direct": bool},
            ...
        ]
    }

Precision counts every flagged sentence that is not a labelled cascade as a
false flag. Every cascade of a synthetic record is a recorded link, so no
flagged sentence is a cascade without a label.

Usage:
    python -m cascading.evaluate --data ../datasets/synthetic/cascades.jsonl \\
        --labels pairs/synthetic/labels.jsonl --detector mention --out eval/mention.json
    python -m cascading.evaluate --data ../datasets/synthetic/cascades.jsonl \\
        --labels pairs/synthetic/labels.jsonl --detector consequence \\
        --features features/cascades --features-after features/cascades-after \\
        --cascade-head heads/cascade --out eval/consequence.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cascading.links import direct, target
from classifier.inputs import read_jsonl

DETECTORS = ("mention", "edited_as_evidence", "staleness_delta", "consequence")
NEEDS = {
    "staleness_delta": ("features", "features_after", "staleness_head"),
    "consequence": ("features", "features_after", "cascade_head"),
    "edited_as_evidence": ("features_evidence", "classifier_head"),
}


class Rows:
    """Pooled sentence vectors of one classifier.extract directory, sliced per article."""

    def __init__(self, directory, layer):
        from cascading.train import first_rows, load_features
        self.ids = json.loads((Path(directory) / "ids.json").read_text())
        self.x, article, _ = load_features(directory, layer)
        self.starts = first_rows(directory)
        self.ends = {self.ids[a]: int(r) + 1 for r, a in enumerate(article)}

    def __getitem__(self, article_id):
        return np.asarray(self.x[self.starts[article_id]:self.ends[article_id]], dtype=np.float32)


def linear_head(path, device):
    """A Linear head saved by classifier.train or staleness.train, with its standardization and layer."""
    import torch
    ckpt = torch.load(path, map_location=device)
    weight = ckpt["state_dict"]["weight"]
    head = torch.nn.Linear(weight.shape[1], weight.shape[0]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    return head.eval(), ckpt["mean"].to(device), ckpt["std"].to(device), ckpt


def build(args, device):
    """The detector, the feature layer it reads, and the classifier head for edited_as_evidence."""
    from cascading.detectors import (ConsequenceDetector, EditedAsEvidenceDetector,
                                     MentionOverlapDetector, StalenessDeltaDetector,
                                     load_consequence_head)
    if args.detector == "mention":
        return MentionOverlapDetector(), None, None
    if args.detector == "staleness_delta":
        head, mean, std, ckpt = linear_head(args.staleness_head, "cpu")
        return StalenessDeltaDetector(head, mean, std, tau2=args.tau2), ckpt["layer"], None
    if args.detector == "consequence":
        head, config = load_consequence_head(args.cascade_head, device=device)
        if args.tau_c is None:
            args.tau_c = config.get("tau_c", 0.5)
        return ConsequenceDetector(head.cpu(), tau_c=args.tau_c), config["layer"], None
    head, mean, std, ckpt = linear_head(args.classifier_head, "cpu")
    return EditedAsEvidenceDetector(), ckpt["layer"], (head, mean, std)


def classifier_probs(head, mean, std, pooled):
    import torch
    with torch.inference_mode():
        return head((torch.from_numpy(pooled) - mean) / std).softmax(-1).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="the dataset the features were extracted from")
    parser.add_argument("--labels", required=True, help="cascade labels JSONL")
    parser.add_argument("--detector", choices=DETECTORS, default="mention")
    parser.add_argument("--features", help="features of the source articles")
    parser.add_argument("--features-after", help="features of the after view")
    parser.add_argument("--features-evidence", help="features of the evidence view")
    parser.add_argument("--cascade-head", help="directory of cascading.train")
    parser.add_argument("--classifier-head", help="head.pt of classifier.train")
    parser.add_argument("--staleness-head", help="head.pt of staleness.train")
    parser.add_argument("--tau2", type=float, default=0.5, help="staleness delta threshold")
    parser.add_argument("--tau-c", type=float, help="consequence threshold; default the head's tuned tau_c")
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args()
    missing = [f"--{n.replace('_', '-')}" for n in NEEDS.get(args.detector, ()) if not getattr(args, n)]
    if missing:
        parser.error(f"--detector {args.detector} needs {' '.join(missing)}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector, layer, classifier = build(args, device)
    before = Rows(args.features, layer) if args.features and layer else None
    after = Rows(args.features_after, layer) if args.features_after and layer else None
    evidence = Rows(args.features_evidence, layer) if args.features_evidence and layer else None

    labels = {row["article_id"]: row for row in read_jsonl(args.labels)}
    groups = {"shared": [0, 0], "unshared": [0, 0]}
    flags = {"flagged": 0, "flagged_cascades": 0, "candidates": 0, "articles": 0}
    for record in read_jsonl(args.data):
        row = labels.get(record["id"])
        if row is None:
            continue
        delta = set(row.get("direct", direct(record)))
        previous = [s.strip() for s in record["source_sentences"]]
        current = [target(record, j) if j in delta else s for j, s in enumerate(previous)]
        inputs = {"previous": previous}
        if before is not None:
            inputs["pooled_before"] = before[record["id"]]
        if after is not None:
            inputs["pooled_after"] = after[record["id"]]
        if evidence is not None:
            inputs["classifier_probs"] = classifier_probs(*classifier, evidence[record["id"]])
        flagged = detector.flag(current, delta, **inputs)

        cascades = [c for c in row["cascade_sentences"] if c["label"] == "cascade"]
        for c in cascades:
            group = groups["shared" if c["shares_mention_with_direct"] else "unshared"]
            group[0] += c["sentence_idx"] in flagged
            group[1] += 1
        flags["flagged"] += len(flagged)
        flags["flagged_cascades"] += len(flagged & {c["sentence_idx"] for c in cascades})
        flags["candidates"] += len(previous) - len(delta)
        flags["articles"] += 1

    def ratio(a, b):
        return a / b if b else None

    results = {
        "detector": args.detector, "tau_c": args.tau_c, "tau2": args.tau2, **flags,
        "group_a_shared_mention": {"cascades": groups["shared"][1], "recall": ratio(*groups["shared"])},
        "group_b_no_shared_mention": {"cascades": groups["unshared"][1], "recall": ratio(*groups["unshared"])},
        "recall": ratio(groups["shared"][0] + groups["unshared"][0], groups["shared"][1] + groups["unshared"][1]),
        "precision": ratio(flags["flagged_cascades"], flags["flagged"]),
        "flag_rate": ratio(flags["flagged"], flags["candidates"]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    def fmt(v):
        return f"{v:.3f}" if v is not None else "-"
    print(f"{args.detector}: {flags['articles']} articles, recall {fmt(results['recall'])}, "
          f"precision {fmt(results['precision'])}, flag rate {fmt(results['flag_rate'])}")
    print(f"  group A (shared mention)    n={groups['shared'][1]}  recall {fmt(results['group_a_shared_mention']['recall'])}")
    print(f"  group B (no shared mention) n={groups['unshared'][1]}  recall {fmt(results['group_b_no_shared_mention']['recall'])}")


if __name__ == "__main__":
    main()
