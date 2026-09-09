"""Test package import and public API exports."""

from __future__ import annotations

import dentaltensor


def test_package_import():
    """Verify package imports cleanly and exposes expected metadata."""
    assert hasattr(dentaltensor, "__version__")
    assert dentaltensor.__version__ == "1.0.0"
    assert dentaltensor.VERSION == "1.0"
    assert dentaltensor.BRAND_NAME == "DentalTensor"
    assert dentaltensor.MODEL_FAMILY == "DentalTensor Vision"
    assert dentaltensor.FULL_MODEL_NAME == "DentalTensor Vision v1.0"
    assert dentaltensor.DEVELOPER == "Nathan Asif"


def test_public_classes_and_thresholds():
    """Verify canonical classes and threshold dictionary are exposed."""
    from dentaltensor import CLASSES, PRODUCTION_THRESHOLDS, DentalTensorVision

    assert len(CLASSES) == 5
    assert "calculus" in CLASSES
    assert "caries" in CLASSES
    assert "gingivitis" in CLASSES
    assert "tooth_discoloration" in CLASSES
    assert "oral_ulcer" in CLASSES

    assert PRODUCTION_THRESHOLDS["calculus"] == 0.35
    assert PRODUCTION_THRESHOLDS["caries"] == 0.55
    assert PRODUCTION_THRESHOLDS["gingivitis"] == 0.50
    assert PRODUCTION_THRESHOLDS["tooth_discoloration"] == 0.65
    assert PRODUCTION_THRESHOLDS["oral_ulcer"] == 0.65

    # Public API class can be instantiated (with mocked/default backend)
    vision = DentalTensorVision()
    assert vision.version == "1.0"
    assert len(vision.classes) == 5
