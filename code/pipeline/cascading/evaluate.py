"""Cascade recall evaluation — Section 9.2.

Measures how many true cascade sentences each detector flags (cascade recall),
broken into two groups per the paper:
  - Group A: cascade sentences that share a mention with a direct edit
             (Detector (a) — mention overlap — can catch these)
  - Group B: cascade sentences with no shared mention
             (only Detector (c) — consequence prediction — can catch these)

Input: a hand-labeled eval file (TSV or JSONL) and an infilling run's
results.jsonl (to obtain the delta_r sets and re-encoded hidden states).

Labeled eval file format (JSONL, one record per article):
    {
        "article_id": str,
        "cascade_sentences": [
            {
                "sentence_idx": int,
                "label": "cascade"|"stylistic",   // hand label
                "shares_mention_with_direct": bool
            },
            ...
        ]
    }

Usage:
    python -m cascading.evaluate \\
        --data datasets/collated/train.jsonl \\
        --labels pairs/eval_labels.jsonl \\
        --features features/train \\
        --detector mention \\
        --out eval/cascade_recall.json

    python -m cascading.evaluate \\
        --data datasets/collated/train.jsonl \\
        --labels pairs/eval_labels.jsonl \\
        --features features/train \\
        --detector consequence \\
        --cascade-head heads/cascade \\
        --out eval/cascade_recall_consequence.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def precision_recall(flagged: set[int], true_cascade: set[int], n_sentences: int) -> dict:
    """Compute precision and recall for cascade detection on one article."""
    tp = len(flagged & true_cascade)
    fp = len(flagged - true_cascade)
    fn = len(true_cascade - flagged)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall    = tp / (tp + fn) if (tp + fn) > 0 else None
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall}


def macro_recall(article_results: list[dict]) -> float | None:
    """Mean recall over articles that have at least one true cascade."""
    vals = [r["recall"] for r in article_results if r["recall"] is not None]
    return float(np.mean(vals)) if vals else None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True,
                        help="collated JSONL (same split used for features)")
    parser.add_argument("--labels", required=True,
                        help="JSONL of hand-labeled cascade sentences")
    parser.add_argument("--features", required=True,
                        help="feature directory from classifier.extract")
    parser.add_argument("--detector", choices=["mention", "edited_as_evidence", "evidence", "staleness_delta", "consequence"],
                        default="mention",
                        help="which cascade detector to evaluate")
    parser.add_argument("--cascade-head", default=None,
                        help="path to trained cascade head dir (for --detector consequence)")
    parser.add_argument("--classifier-head", default=None,
                        help="path to trained classifier head dir (for --detector edited_as_evidence)")
    parser.add_argument("--staleness-head", default=None,
                        help="path to staleness head .pt (for --detector staleness_delta)")
    parser.add_argument("--tau2", type=float, default=0.5,
                        help="staleness delta threshold τ_2")
    parser.add_argument("--tau-c", type=float, default=0.5,
                        help="consequence detection threshold τ_c")
    parser.add_argument("--tau-d", type=float, default=0.45,
                        help="classifier DROP/EDIT threshold τ_d (for edited_as_evidence)")
    parser.add_argument("--out", required=True,
                        help="output JSON path for evaluation results")
    args = parser.parse_args()

    import torch
    from cascading.detectors import (
        ConsequenceDetector,
        EditedAsEvidenceDetector,
        MentionOverlapDetector,
        StalenessDeltaDetector,
        load_consequence_head,
    )
    from cascading.train import load_features

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load features
    print(f"Loading features from {args.features}...")
    features, article_arr, labels_arr = load_features(args.features)
    ids_list = json.loads((Path(args.features) / "ids.json").read_text())

    # Build article_id → row slice mapping
    art_to_rows: dict[str, tuple[int, int]] = {}
    cur_art, start = None, 0
    for row, art_i in enumerate(article_arr):
        aid = ids_list[art_i]
        if aid != cur_art:
            if cur_art is not None:
                art_to_rows[cur_art] = (start, row)
            cur_art, start = aid, row
    if cur_art is not None:
        art_to_rows[cur_art] = (start, len(article_arr))

    # Build data records index
    data_records = {r["id"]: r for r in load_jsonl(args.data)}

    # Load labeled eval set
    eval_labels = load_jsonl(args.labels)

    # Build detector
    if args.detector == "mention":
        detector = MentionOverlapDetector()
    elif args.detector in ("edited_as_evidence", "evidence"):
        cls_head = None
        cls_mean, cls_std = None, None
        if args.classifier_head:
            ckpt = torch.load(Path(args.classifier_head) / "head.pt", map_location=device)
            w = ckpt["state_dict"]["weight"]
            cls_head = torch.nn.Linear(w.shape[1], w.shape[0]).to(device)
            cls_head.load_state_dict(ckpt["state_dict"])
            cls_head.eval()
            cls_mean = ckpt.get("mean")
            cls_std = ckpt.get("std")
        detector = EditedAsEvidenceDetector(
            classifier_head=cls_head,
            classifier_mean=cls_mean,
            classifier_std=cls_std,
            tau_d=args.tau_d,
        )
    elif args.detector == "staleness_delta":
        if not args.staleness_head:
            raise SystemExit("--staleness-head required for staleness_delta detector")
        ckpt = torch.load(args.staleness_head, map_location=device)
        head = torch.nn.Linear(ckpt["state_dict"]["weight"].shape[1], 1).to(device)
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        detector = StalenessDeltaDetector(
            staleness_head=head,
            mean=ckpt["mean"].to(device),
            std=ckpt["std"].to(device),
            tau2=args.tau2,
        )
    else:  # consequence
        if not args.cascade_head:
            raise SystemExit("--cascade-head required for consequence detector")
        cascade_head, _ = load_consequence_head(args.cascade_head, device=device)
        detector = ConsequenceDetector(head=cascade_head, tau_c=args.tau_c)

    # Evaluate per article
    group_a: list[dict] = []   # share mention with direct edit
    group_b: list[dict] = []   # no shared mention

    for article_eval in eval_labels:
        aid = article_eval["article_id"]
        if aid not in art_to_rows or aid not in data_records:
            continue

        r0, r1 = art_to_rows[aid]
        record = data_records[aid]
        sentences = record["source_sentences"]
        labels_raw = record["labels"]
        n = len(sentences)

        # delta_r = all EDIT and DROP sentences (oracle labels for the eval)
        delta_r = {i for i, lbl in enumerate(labels_raw)
                   if lbl.get("label") in ("EDIT", "DROP")}

        pooled = features[r0:r1]   # [n_sents, d]
        # Use same pooled for before/after (no actual repair state available for eval)
        flagged = detector.flag(
            sentences=sentences,
            delta_r=delta_r,
            pooled_before=pooled,
            pooled_after=pooled,
        )

        # Score against labeled cascades
        for entry in article_eval["cascade_sentences"]:
            if entry["label"] != "cascade":
                continue
            sidx = entry["sentence_idx"]
            true_set = {sidx}
            result = precision_recall(flagged, true_set, n)
            result["article_id"] = aid
            result["sentence_idx"] = sidx
            result["shares_mention"] = entry["shares_mention_with_direct"]
            if entry["shares_mention_with_direct"]:
                group_a.append(result)
            else:
                group_b.append(result)

    # Aggregate
    def agg(group: list[dict]) -> dict:
        return {
            "n_sentences": len(group),
            "cascade_recall": macro_recall(group),
            "tp_total": sum(r["tp"] for r in group),
            "fn_total": sum(r["fn"] for r in group),
            "fp_total": sum(r["fp"] for r in group),
        }

    results = {
        "detector": args.detector,
        "tau_c": args.tau_c,
        "tau2": args.tau2,
        "group_a_shared_mention": agg(group_a),
        "group_b_no_shared_mention": agg(group_b),
        "overall": agg(group_a + group_b),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Detector: {args.detector}")
    print(f"Group A (shared mention):   n={results['group_a_shared_mention']['n_sentences']}  "
          f"cascade recall={results['group_a_shared_mention']['cascade_recall']}")
    print(f"Group B (no shared mention):n={results['group_b_no_shared_mention']['n_sentences']}  "
          f"cascade recall={results['group_b_no_shared_mention']['cascade_recall']}")
    print(f"Overall:                    n={results['overall']['n_sentences']}  "
          f"cascade recall={results['overall']['cascade_recall']}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
