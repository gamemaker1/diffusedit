"""Cascading edit detection for DiffusEdit (Sections 6.4, 7.3, 8.5).

This package implements all four cascade detectors described in the design:
  (a) MentionOverlapDetector: flags by entity and digit string overlap
  (b) EditedAsEvidenceDetector: appends edited sentences to evidence in round r+1
  (c) StalenessDeltaDetector: flags by relative staleness score rank rise
  (d) ConsequenceDetector: trained pairwise MLP consequence prediction head
"""

from cascading.detectors import (
    ConsequenceDetector,
    EditedAsEvidenceDetector,
    MentionOverlapDetector,
    StalenessDeltaDetector,
    build_detector,
)
from cascading.mentions import Extractor, get_extractor

