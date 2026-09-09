"""Centralized confidence thresholds and class normalization for DentalTensor Vision."""

from __future__ import annotations

from typing import Dict, Optional

# Locked production calibrated thresholds
PRODUCTION_THRESHOLDS: Dict[str, float] = {
    "calculus": 0.35,
    "caries": 0.55,
    "gingivitis": 0.50,
    "tooth_discoloration": 0.65,
    "oral_ulcer": 0.65,
}

GLOBAL_FALLBACK_THRESHOLD: float = 0.50

# Normalization mapping from various naming conventions (e.g. YOLO labels, internal codes)
# to canonical public DentalTensor finding classes
CLASS_NAME_ALIASES: Dict[str, str] = {
    # Calculus / Tartar
    "calculus": "calculus",
    "tartar": "calculus",
    "dental_calculus": "calculus",
    # Caries / Cavity
    "caries": "caries",
    "cavity": "caries",
    "cavity_suspect": "caries",
    "decay": "caries",
    # Gingivitis
    "gingivitis": "gingivitis",
    "gingivitis_signs": "gingivitis",
    # Tooth Discoloration
    "tooth discoloration": "tooth_discoloration",
    "tooth_discoloration": "tooth_discoloration",
    "discoloration": "tooth_discoloration",
    "staining": "tooth_discoloration",
    # Oral Ulcer
    "ulcer": "oral_ulcer",
    "oral_ulcer": "oral_ulcer",
    "aphthous_ulcer": "oral_ulcer",
}

# Raw YOLO11 integer index to canonical class name mapping
YOLO_INDEX_TO_CLASS: Dict[int, str] = {
    0: "calculus",
    1: "caries",
    2: "gingivitis",
    3: "tooth_discoloration",
    4: "oral_ulcer",
}


def normalize_class_name(raw_name: str) -> str:
    """Normalize a raw class label or alias into the canonical DentalTensor class name."""
    cleaned = raw_name.strip().lower()
    return CLASS_NAME_ALIASES.get(cleaned, cleaned.replace(" ", "_").replace("-", "_"))


def get_confidence_threshold(
    class_name: str, override: Optional[float] = None
) -> float:
    """Return the applicable confidence threshold for a class.

    If `override` is explicitly provided, it takes precedence over class-specific cutoffs.
    Otherwise, returns the calibrated production threshold, falling back to 0.50.
    """
    if override is not None:
        return float(override)
    normalized = normalize_class_name(class_name)
    return PRODUCTION_THRESHOLDS.get(normalized, GLOBAL_FALLBACK_THRESHOLD)


def get_all_thresholds(override: Optional[float] = None) -> Dict[str, float]:
    """Return dictionary of all class thresholds with optional override applied."""
    if override is not None:
        return {cls: float(override) for cls in PRODUCTION_THRESHOLDS}
    return dict(PRODUCTION_THRESHOLDS)
