"""Test confidence thresholds and class name normalization."""

from __future__ import annotations

from dentaltensor.config import GLOBAL_FALLBACK_THRESHOLD, PRODUCTION_THRESHOLDS
from dentaltensor.inference.thresholds import (
    YOLO_INDEX_TO_CLASS,
    get_all_thresholds,
    get_confidence_threshold,
    normalize_class_name,
)


def test_production_threshold_values():
    """Ensure exact locked calibrated threshold values."""
    assert PRODUCTION_THRESHOLDS["calculus"] == 0.35
    assert PRODUCTION_THRESHOLDS["caries"] == 0.55
    assert PRODUCTION_THRESHOLDS["gingivitis"] == 0.50
    assert PRODUCTION_THRESHOLDS["tooth_discoloration"] == 0.65
    assert PRODUCTION_THRESHOLDS["oral_ulcer"] == 0.65
    assert GLOBAL_FALLBACK_THRESHOLD == 0.50


def test_class_normalization_aliases():
    """Verify alias mapping to canonical names."""
    assert normalize_class_name("calculus") == "calculus"
    assert normalize_class_name("tartar") == "calculus"

    assert normalize_class_name("caries") == "caries"
    assert normalize_class_name("cavity") == "caries"
    assert normalize_class_name("cavity_suspect") == "caries"

    assert normalize_class_name("gingivitis") == "gingivitis"
    assert normalize_class_name("gingivitis_signs") == "gingivitis"

    assert normalize_class_name("tooth discoloration") == "tooth_discoloration"
    assert normalize_class_name("tooth_discoloration") == "tooth_discoloration"
    assert normalize_class_name("discoloration") == "tooth_discoloration"

    assert normalize_class_name("ulcer") == "oral_ulcer"
    assert normalize_class_name("oral_ulcer") == "oral_ulcer"
    assert normalize_class_name("aphthous_ulcer") == "oral_ulcer"


def test_yolo_index_mapping():
    """Verify YOLO integer class indices map to the 5 canonical classes."""
    assert YOLO_INDEX_TO_CLASS[0] == "calculus"
    assert YOLO_INDEX_TO_CLASS[1] == "caries"
    assert YOLO_INDEX_TO_CLASS[2] == "gingivitis"
    assert YOLO_INDEX_TO_CLASS[3] == "tooth_discoloration"
    assert YOLO_INDEX_TO_CLASS[4] == "oral_ulcer"


def test_get_confidence_threshold_and_override():
    """Verify threshold retrieval with and without overrides."""
    # Without override -> production threshold
    assert get_confidence_threshold("calculus") == 0.35
    assert get_confidence_threshold("tartar") == 0.35
    assert get_confidence_threshold("caries") == 0.55
    assert get_confidence_threshold("tooth discoloration") == 0.65
    assert get_confidence_threshold("ulcer") == 0.65
    assert get_confidence_threshold("non_existent") == 0.50

    # With override -> explicit value everywhere
    assert get_confidence_threshold("calculus", override=0.80) == 0.80
    assert get_confidence_threshold("caries", override=0.80) == 0.80


def test_get_all_thresholds():
    """Verify all thresholds dict returned."""
    thresh = get_all_thresholds()
    assert len(thresh) == 5
    assert thresh["calculus"] == 0.35

    override_thresh = get_all_thresholds(override=0.72)
    assert all(v == 0.72 for v in override_thresh.values())
