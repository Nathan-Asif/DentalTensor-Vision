"""Post-processing and filtering for DentalTensor Vision detections."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dentaltensor.inference.thresholds import get_confidence_threshold, normalize_class_name
from dentaltensor.schemas import DentalFinding, DentalVisionResult, ImageMetadata


def filter_and_format_findings(
    raw_detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    threshold_override: Optional[float] = None,
) -> List[DentalFinding]:
    """Filter raw YOLO detections using calibrated class thresholds or a global override.

    Parameters:
    - raw_detections: List of dicts with keys: 'class_name' (or 'cls_idx'), 'confidence', 'bbox' [x1, y1, x2, y2]
    - image_width: Image width in pixels
    - image_height: Image height in pixels
    - threshold_override: Optional explicit threshold override (e.g. from CLI or API parameter)

    Returns:
    - List of validated DentalFinding objects
    """
    findings: List[DentalFinding] = []

    for det in raw_detections:
        raw_name = str(det.get("class_name", ""))
        conf = float(det.get("confidence", 0.0))
        bbox = det.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        canonical_name = normalize_class_name(raw_name)
        threshold_used = get_confidence_threshold(
            canonical_name, override=threshold_override
        )

        if conf < threshold_used:
            continue

        x1, y1, x2, y2 = [float(c) for c in bbox]
        # Clamp coordinates to image boundaries
        x1_clamped = max(0.0, min(float(image_width), x1))
        y1_clamped = max(0.0, min(float(image_height), y1))
        x2_clamped = max(0.0, min(float(image_width), x2))
        y2_clamped = max(0.0, min(float(image_height), y2))

        # Compute normalized coordinates
        w = max(1.0, float(image_width))
        h = max(1.0, float(image_height))
        norm_box = (
            round(x1_clamped / w, 4),
            round(y1_clamped / h, 4),
            round(x2_clamped / w, 4),
            round(y2_clamped / h, 4),
        )

        finding = DentalFinding(
            class_name=canonical_name,
            confidence=round(conf, 4),
            bbox=(
                round(x1_clamped, 2),
                round(y1_clamped, 2),
                round(x2_clamped, 2),
                round(y2_clamped, 2),
            ),
            threshold_used=round(threshold_used, 4),
            bbox_normalized=norm_box,
        )
        findings.append(finding)

    # Deterministic priority ordering: caries > oral_ulcer > calculus > gingivitis > tooth_discoloration
    priority_order = {
        "caries": 0,
        "oral_ulcer": 1,
        "calculus": 2,
        "gingivitis": 3,
        "tooth_discoloration": 4,
    }
    findings.sort(
        key=lambda f: (priority_order.get(f.class_name, 99), -f.confidence)
    )

    return findings


def build_vision_result(
    findings: List[DentalFinding],
    image_width: int,
    image_height: int,
    inference_ms: Optional[int] = None,
    source_identifier: Optional[str] = None,
) -> DentalVisionResult:
    """Construct structured DentalVisionResult from findings and image metadata."""
    metadata = ImageMetadata(
        width=image_width,
        height=image_height,
        channels=3,
        format="RGB",
        source=source_identifier,
    )
    return DentalVisionResult(
        model="DentalTensor Vision",
        version="1.0",
        findings=findings,
        image_metadata=metadata,
        inference_ms=inference_ms,
    )
