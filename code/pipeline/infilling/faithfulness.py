"""AlignScore of a run's updated sentences against the evidence (Section 9.2).

For one article, U(output) is the output sentences that do not occur
verbatim, after whitespace normalization, in the source article. AlignScore
reads the article's evidence snippets as the context and the concatenation of
U(output) as the claim. It splits the context into chunks and the claim into
sentences, scores each claim sentence against the chunks, and aggregates the
results into one number in [0, 1]. An article with an empty U(output) gets
no score.

The evidence is always the record's evidence, so a --no-evidence run is
scored against the evidence it did not see.

AlignScore pins PyTorch 1.12, which conflicts with the pipeline's torch, so
this module runs in its own environment. It imports alignscore only inside
main(), and infilling.metrics imports updated() from here without it.

The scores go to <run>/alignscore.jsonl, one {"id", "score"} line per
article, with a null score when U(output) is empty. A rerun resumes after
the last article.

Usage:
    python -m infilling.faithfulness runs/fix-oracle ../datasets/collated/train.jsonl \\
        --ckpt checkpoints/AlignScore-large.ckpt
"""

import argparse
import json
import re
from pathlib import Path

from classifier.inputs import read_jsonl


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def updated(sentences, source):
    """Returns U(sentences), the non-empty sentences absent from `source`, a set of normalized sentences."""
    return [s.strip() for s in sentences if s.strip() and norm(s) not in source]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("data", help="the collated split the run read")
    parser.add_argument("--ckpt", required=True, help="AlignScore-large.ckpt or AlignScore-base.ckpt")
    parser.add_argument("--model", default="roberta-large", help="roberta-base for the base checkpoint")
    parser.add_argument("--mode", default="nli_sp", choices=["nli_sp", "nli", "bin_sp", "bin"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=64, help="articles per scoring call")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from alignscore import AlignScore

    run = Path(args.run)
    out = run / "alignscore.jsonl"
    done = {json.loads(line)["id"] for line in open(out) if line.strip()} if out.exists() else set()
    results = [r for r in read_jsonl(run / "results.jsonl") if r["id"] not in done]
    wanted = {r["id"] for r in results}
    records = {r["id"]: r for r in read_jsonl(args.data) if r["id"] in wanted}
    scorer = AlignScore(model=args.model, batch_size=args.batch, device=args.device,
                        ckpt_path=args.ckpt, evaluation_mode=args.mode)
    print(f"{len(results)} articles to score, {len(done)} already done")

    with open(out, "a") as f:
        for start in range(0, len(results), args.chunk):
            lines, contexts, claims = [], [], []
            for result in results[start:start + args.chunk]:
                record = records[result["id"]]
                claim = updated(result["output"], {norm(s) for s in record["source_sentences"]})
                context = "\n".join(f"{s['title']}: {s['text']}" for s in record["evidence"])
                lines.append({"id": result["id"], "score": None})
                if claim and context:
                    contexts.append(context)
                    claims.append(" ".join(claim))
                    lines[-1]["score"] = len(claims) - 1
            scores = scorer.score(contexts=contexts, claims=claims) if claims else []
            for line in lines:
                if line["score"] is not None:
                    line["score"] = float(scores[line["score"]])
                f.write(json.dumps(line) + "\n")
            f.flush()
            print(f"{start + len(lines)}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
