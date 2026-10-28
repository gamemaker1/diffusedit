"""Cascading edit detection for DiffusEdit (Sections 6.4, 7.3, 8.5).

This package implements the three cascade detectors of the design, lettered
as in the paper, and the progress report's addition as (d):
  (a) MentionOverlapDetector: flags by entity and digit string overlap
  (b) StalenessDeltaDetector: flags by relative staleness score rank rise
  (c) ConsequenceDetector: trained pairwise MLP consequence prediction head
  (d) EditedAsEvidenceDetector: adds edited sentences to the evidence in round r+1
"""

from cascading.detectors import (
    ConsequenceDetector,
    EditedAsEvidenceDetector,
    MentionOverlapDetector,
    StalenessDeltaDetector,
    build_detector,
)
from cascading.mentions import Extractor, get_extractor

