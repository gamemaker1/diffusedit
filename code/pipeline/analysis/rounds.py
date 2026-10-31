"""The article after each round of a loop run, as one run directory per round.

infilling.loop logs only the final article, but every edit records its round
and the chosen candidate holds the whole sentence, so the article after round
r is the source with the edits and deletions of rounds 1 to r applied. The
article rebuilt from every round must equal the logged output, and the script
stops if it does not.

Each round directory holds results.jsonl in the format of infilling.metrics,
with the "rounds" field cut to the first r rounds so that cascade recall
counts only the flags raised so far.

Usage:
    python -m analysis.rounds outputs/jl/full/runs/loop-consequence OUT_DIR
"""

import argparse
import json
from pathlib import Path

from classifier.inputs import read_jsonl


def after_round(result, r):
    """The output sentences after the first r rounds."""
    sentences = list(result["source"])
    # A DROP sentence of the first pass is deleted before round 1 and appears
    # in no round, so it is read off the output: empty there, and never
    # repaired or deleted by a round.
    touched = {j for entry in result["rounds"] for j in entry["repaired"] + entry["deleted"]}
    touched |= {edit["sentence"] for edit in result["edits"]}
    for j, text in enumerate(result["output"]):
        if text == "" and j not in touched:
            sentences[j] = ""
    for entry in result["rounds"][:r]:
        for edit in result["edits"]:
            if edit["round"] != entry["round"]:
                continue
            decision = edit["decisions"][-1]
            if decision["chosen"] is not None:
                sentences[edit["sentence"]] = decision["candidates"][decision["chosen"]]["text"]
        for j in entry["deleted"]:
            sentences[j] = ""
    return sentences


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("out")
    args = parser.parse_args()
    results = list(read_jsonl(Path(args.run) / "results.jsonl"))
    most = max(len(r["rounds"]) for r in results)
    for result in results:
        rebuilt = after_round(result, len(result["rounds"]))
        if rebuilt != result["output"]:
            raise SystemExit(f"{result['id']}: the rebuilt article differs from the logged output")
    for r in range(1, most + 1):
        out = Path(args.out) / f"round{r}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "results.jsonl", "w") as f:
            for result in results:
                f.write(json.dumps({"id": result["id"], "output": after_round(result, r),
                                    "rounds": result["rounds"][:r]}) + "\n")
        print(f"round {r}: {sum(len(x['rounds']) >= r for x in results)} articles reached it")


if __name__ == "__main__":
    main()
