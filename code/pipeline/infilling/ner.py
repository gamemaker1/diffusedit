"""Named entities of every article's evidence and source sentences, for the
evidence-copy candidates of infilling.repair.

spaCy runs here, in the evaluation environment, and not in the GPU job, whose
environment has no spaCy. The GPU job reads the JSON this writes next to the
data, so split/val.jsonl gives split/val.entities.json, holding per article id

    evidence   [text, label] of every entity in the evidence snippets
    sentences  per source sentence, [text, label] of its entities

    python -m infilling.ner data/split/val.jsonl data/split/test.jsonl
"""

import json
import sys
from pathlib import Path


def path_of(data):
    """The entity file of a data file."""
    data = Path(data)
    return data.with_name(data.stem + ".entities.json")


def main():
    import spacy  # here, so that the GPU job can import path_of without spaCy

    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    for data in sys.argv[1:]:
        out = {}
        with open(data) as f:
            for line in f:
                record = json.loads(line)
                evidence = [f"{s['title']}: {s['text']}" for s in record["evidence"]]
                out[record["id"]] = {
                    "evidence": [[e.text, e.label_] for doc in nlp.pipe(evidence) for e in doc.ents],
                    "sentences": [[[e.text, e.label_] for e in doc.ents]
                                  for doc in nlp.pipe(record["source_sentences"])],
                }
        path_of(data).write_text(json.dumps(out))
        print(f"{data}: {len(out)} articles, "
              f"{sum(len(v['evidence']) for v in out.values())} evidence entities -> {path_of(data)}")


if __name__ == "__main__":
    main()
