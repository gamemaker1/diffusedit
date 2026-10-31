"""Verifier support and entity scores of the output sentences of several runs.

For every gold EDIT sentence of the articles in --ids, the output of each run
is scored by the VitaminC verifier of the ranking step: the highest
probability of SUPPORTS over the article's evidence passages, written as
"title: text" as infilling.run does. Copy scores the unchanged source. The
mean per run is a ratio of sums over articles with a 95% bootstrap interval.

Usage:
    python -m analysis.support DATA name=RUN_DIR ... --ids IDS --out support.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.extra import ratio
from classifier.inputs import read_jsonl
from infilling.score import VERIFIER, Verifier


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("runs", nargs="+", help="name=run_dir, or copy")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    keep = set(json.loads(Path(args.ids).read_text()))
    records = {r["id"]: r for r in read_jsonl(args.data) if r["id"] in keep}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verifier = Verifier(VERIFIER, device, batch=64)
    out = {}
    for item in args.runs:
        name, path = (item, None) if "=" not in item else item.split("=", 1)
        outputs = {i: r["source_sentences"] for i, r in records.items()} if path is None else \
            {r["id"]: r["output"] for r in read_jsonl(Path(path) / "results.jsonl") if r["id"] in records}
        rows = {}
        for i, record in records.items():
            if i not in outputs:
                continue
            evidence = [f"{s['title']}: {s['text']}" for s in record["evidence"]]
            js = [j for j, e in enumerate(record["labels"]) if e.get("label") == "EDIT"]
            claims = [outputs[i][j] for j in js]
            scores = verifier([c if c.strip() else "." for c in claims], evidence)
            rows[i] = [{"support": s} for s in scores]
        m, lo, hi = ratio(rows, "support")
        out[name] = {"support": m, "low": lo, "high": hi, "sentences": sum(len(v) for v in rows.values())}
        print(f"{name:12s} support {m:.3f} [{lo:.3f}, {hi:.3f}]  n={out[name]['sentences']}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
