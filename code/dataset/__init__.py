"""Synthetic data generation package for DiffusEdit."""

from .schema import EvidenceSnippet, SentenceLabel, SyntheticRecord, read_jsonl, write_jsonl

__all__ = ["EvidenceSnippet", "SentenceLabel", "SyntheticRecord", "read_jsonl", "write_jsonl"]
