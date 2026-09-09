"""Test detection postprocessing and filtering logic."""

from __future__ import annotations

from dentaltensor.inference.postprocessing import (
    build_vision_result,
    filter_and_format_findings,
)


def test_postprocessing_threshold_filtering():
    """Verify raw detections are filtered strictly by class-calibrated thresholds."""
    raw = [
        # calculus threshold is 0.35 -> 0.40 kept, 0.30 dropped
        {"class_name": "calculus", "confidence": 0.40, "bbox": [10, 10, 50, 50]},
        {"class_name": "calculus", "confidence": 0.30, "bbox": [10, 10, 50, 50]},
        # caries threshold is 0.55 -> 0.50 dropped, 0.70 kept
        {"class_name": "caries", "confidence": 0.50, "bbox": [20, 20, 60, 60]},
        {"class_name": "caries", "confidence": 0.70, "bbox": [30, 30, 70, 70]},
        # tooth discoloration threshold is 0.65 -> 0.60 dropped, 0.80 kept
        {"class_name": "tooth discoloration", "confidence": 0.60, "bbox": [40, 40, 80, 80]},
        {"class_name": "tooth discoloration", "confidence": 0.80, "bbox": [50, 50, 90, 90]},
    ]

    findings = filter_and_format_findings(
        raw_detections=raw,
        image_width=100,
        image_height=100,
    )

    # Kept: calculus (0.40), caries (0.70), tooth_discoloration (0.80)
    assert len(findings) == 3

    class_names = [f.class_name for f in findings]
    # Sorted by priority: caries before calculus before tooth_discoloration
    assert class_names == ["caries", "calculus", "tooth_discoloration"]

    # Check threshold_used recorded
    caries_f = next(f for f in findings if f.class_name == "caries")
    assert caries_f.confidence == 0.70
    assert caries_f.threshold_used == 0.55

    calc_f = next(f for f in findings if f.class_name == "calculus")
    assert calc_f.confidence == 0.40
    assert calc_f.threshold_used == 0.35


def test_postprocessing_global_threshold_override():
    """Verify global threshold override takes precedence over class-calibrated values."""
    raw = [
        {"class_name": "calculus", "confidence": 0.30, "bbox": [10, 10, 50, 50]},
        {"class_name": "caries", "confidence": 0.45, "bbox": [20, 20, 60, 60]},
    ]

    # With override=0.25, both 0.30 and 0.45 should pass
    findings = filter_and_format_findings(
        raw_detections=raw,
        image_width=100,
        image_height=100,
        threshold_override=0.25,
    )
    assert len(findings) == 2
    for f in findings:
        assert f.threshold_used == 0.25


def test_normalized_bounding_box_computation():
    """Verify pixel coordinates are normalized correctly to [0.0, 1.0]."""
    raw = [
        {"class_name": "caries", "confidence": 0.90, "bbox": [50, 100, 150, 300]}
    ]
    findings = filter_and_format_findings(
        raw_detections=raw,
        image_width=200,
        image_height=400,
    )
    assert len(findings) == 1
    f = findings[0]
    # x1=50/200=0.25, y1=100/400=0.25, x2=150/200=0.75, y2=300/400=0.75
    assert f.bbox_normalized == (0.25, 0.25, 0.75, 0.75)


def test_build_vision_result():
    """Verify build_vision_result creates complete structured result."""
    findings = filter_and_format_findings(
        raw_detections=[{"class_name": "caries", "confidence": 0.90, "bbox": [10, 10, 50, 50]}],
        image_width=100,
        image_height=100,
    )
    result = build_vision_result(findings, image_width=100, image_height=100, inference_ms=35)
    assert result.model == "DentalTensor Vision"
    assert result.version == "1.0"
    assert result.inference_ms == 35
    assert result.image_metadata is not None
    assert result.image_metadata.width == 100
    assert result.image_metadata.height == 100
    assert len(result.findings) == 1
