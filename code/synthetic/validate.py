"""Validation and integrity check for synthetic datasets.

Verifies:
1. Structural schema compliance with code/synthetic/schema.py.
2. Character boundary correctness: each stale_span [a, b] is within bounds.
3. Compatibility with classifier.inputs (tokenization and stale token marking).
4. Label distribution statistics (KEEP / EDIT / DROP ratios).

Usage:
    python -m code.synthetic.validate datasets/synthetic/train.jsonl [--check-tokens]
"""

import argparse
from pathlib import Path
import sys

# Ensure both package and direct script execution work cleanly
_curr = Path(__file__).resolve().parent
if str(_curr.parent) not in sys.path:
    sys.path.insert(0, str(_curr.parent))
if str(_curr) not in sys.path:
    sys.path.insert(0, str(_curr))

try:
    from .schema import read_jsonl
except ImportError:
    from schema import read_jsonl



def validate_file(path: str | Path, check_tokens: bool = False, model_name: str | None = None) -> bool:
    print(f"=== Validating {path} ===")
    records = read_jsonl(path)
    print(f"Loaded {len(records)} records.")

    total_sentences = 0
    label_counts = {"KEEP": 0, "EDIT": 0, "DROP": 0}
    total_spans = 0
    span_errors = 0

    for rec in records:
        valid, err = rec.validate()
        if not valid:
            print(f"ERROR: {err}")
            return False

        for i, (sent, lbl) in enumerate(zip(rec.source_sentences, rec.labels)):
            total_sentences += 1
            label_counts[lbl.label] += 1
            stripped = sent.strip()

            if lbl.label == "EDIT":
                total_spans += len(lbl.stale_spans)
                for start, end in lbl.stale_spans:
                    if not (0 <= start < end <= len(stripped)):
                        print(f"ERROR in {rec.id} sent {i}: span [{start}, {end}] out of bounds (len={len(stripped)})")
                        span_errors += 1
                    else:
                        stale_text = stripped[start:end]
                        if not stale_text.strip():
                            print(f"WARNING in {rec.id} sent {i}: span [{start}, {end}] is whitespace only: {stale_text!r}")

    print(f"\n--- Statistics ---")
    print(f"Articles: {len(records)}")
    print(f"Total Sentences: {total_sentences}")
    for k, v in label_counts.items():
        print(f"  {k:5s}: {v:6d} ({v / total_sentences:6.2%})")
    print(f"Total Stale Spans: {total_spans} (errors: {span_errors})")

    if span_errors > 0:
        return False

    if check_tokens:
        print("\n--- Testing with classifier.inputs and AutoTokenizer ---")
        try:
            from transformers import AutoTokenizer
            from classifier.inputs import build, MODEL

            model_id = model_name or MODEL
            print(f"Loading tokenizer: {model_id}...")
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

            tested = min(50, len(records))
            for i in range(tested):
                example = build(records[i].to_dict(), tok)
                assert len(example.spans) == len(records[i].source_sentences)
                assert len(example.labels) == len(records[i].labels)
            print(f"Successfully validated {tested} records against classifier.inputs.build()! 0 token errors.")
        except Exception as e:
            print(f"Token check failed or skipped: {e}")
            return False

    print("\n[PASSED] All checks passed successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="path to synthetic jsonl file")
    parser.add_argument("--check-tokens", action="store_true", help="run tokenization and span checks with LLaDA tokenizer")
    parser.add_argument("--model", default=None, help="custom tokenizer model name")
    args = parser.parse_args()

    success = validate_file(args.data, check_tokens=args.check_tokens, model_name=args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
