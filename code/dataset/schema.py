"""Data structures and validation for synthetic DiffusEdit records.

Ensures strict compatibility with the schema expected by classifier.inputs:
Each record is a JSON object with:
    id: str
    evidence: list[{"title": str, "text": str}]
    source_sentences: list[str]
    labels: list[{"label": "KEEP"|"EDIT"|"DROP", "stale_spans": [[start, end], ...]}]
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

VALID_LABELS = {"KEEP", "EDIT", "DROP"}


@dataclass
class EvidenceSnippet:
    title: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "text": self.text}


@dataclass
class SentenceLabel:
    label: str  # KEEP, EDIT, DROP
    stale_spans: list[tuple[int, int]] = field(default_factory=list)
    target: str = ""  # Gold target rewrite for evaluate.py / ROUGE scoring

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "stale_spans": [list(span) for span in self.stale_spans],
            "target": self.target,
        }


@dataclass
class SyntheticRecord:
    id: str
    evidence: list[EvidenceSnippet]
    source_sentences: list[str]
    labels: list[SentenceLabel]
    target_sentences: list[str] = field(default_factory=list)
    # One {"direct": j, "cascade": k, "kind": ...} link per cascade sentence k,
    # which must change only because sentence j changed. One j may have several.
    cascades: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        targets = self.target_sentences or [l.target for l in self.labels if l.label != "DROP"]
        return {
            "id": self.id,
            "evidence": [e.to_dict() for e in self.evidence],
            "source_sentences": list(self.source_sentences),
            "target": " ".join(targets),
            "target_sentences": list(targets),
            "labels": [l.to_dict() for l in self.labels],
            "cascades": [dict(c) for c in self.cascades],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SyntheticRecord":
        evidence = [EvidenceSnippet(e["title"], e["text"]) for e in data.get("evidence", [])]
        labels = [
            SentenceLabel(
                label=l["label"],
                stale_spans=[tuple(s) for s in l.get("stale_spans", [])],
                target=l.get("target", ""),
            )
            for l in data.get("labels", [])
        ]
        return cls(
            id=data["id"],
            evidence=evidence,
            source_sentences=data["source_sentences"],
            labels=labels,
            target_sentences=data.get("target_sentences", []),
            cascades=data.get("cascades", []),
        )

    def validate(self) -> tuple[bool, str | None]:
        """Validate internal consistency and alignment."""
        if not self.id:
            return False, "Missing or empty record ID"
        if not self.source_sentences:
            return False, f"{self.id}: source_sentences cannot be empty"
        if len(self.source_sentences) != len(self.labels):
            return False, (
                f"{self.id}: mismatch between {len(self.source_sentences)} sentences "
                f"and {len(self.labels)} labels"
            )

        for i, (sent, lbl) in enumerate(zip(self.source_sentences, self.labels)):
            if not isinstance(sent, str):
                return False, f"{self.id} sentence {i}: not a string"
            if lbl.label not in VALID_LABELS:
                return False, f"{self.id} sentence {i}: invalid label {lbl.label}"

            sent_len = len(sent.strip())
            if lbl.label == "EDIT":
                if not lbl.stale_spans:
                    return False, f"{self.id} sentence {i}: EDIT label has empty stale_spans"
                for start, end in lbl.stale_spans:
                    if not (0 <= start < end <= sent_len):
                        return False, (
                            f"{self.id} sentence {i}: stale_span [{start}, {end}] out of "
                            f"bounds for stripped sentence length {sent_len}"
                        )
            elif lbl.label in ("KEEP", "DROP"):
                if lbl.stale_spans:
                    return False, (
                        f"{self.id} sentence {i}: {lbl.label} must have empty stale_spans, "
                        f"got {lbl.stale_spans}"
                    )

        return True, None


def write_jsonl(records: list[SyntheticRecord], output_path: str | Path) -> None:
    """Write records to a JSONL file after validation."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            valid, err = r.validate()
            if not valid:
                raise ValueError(f"Record validation failed: {err}")
            f.write(r.to_json() + "\n")


def read_jsonl(path: str | Path) -> list[SyntheticRecord]:
    """Read and validate records from a JSONL file."""
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rec = SyntheticRecord.from_dict(data)
            valid, err = rec.validate()
            if not valid:
                raise ValueError(f"Line {line_num} validation failed: {err}")
            records.append(rec)
    return records
