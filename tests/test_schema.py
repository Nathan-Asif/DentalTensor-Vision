"""Test Pydantic output schemas and JSON serialization."""

from __future__ import annotations

import json
from dentaltensor.schemas import DentalFinding, DentalVisionResult, ImageMetadata


def test_schema_serialization_format():
    """Verify DentalVisionResult serialization matches the canonical output format."""
    finding = DentalFinding(
        class_name="caries",
        confidence=0.71,
        bbox=(120.0, 80.0, 240.0, 190.0),
        threshold_used=0.55,
        bbox_normalized=(0.1875, 0.1667, 0.375, 0.3958),
    )
    metadata = ImageMetadata(
        width=640,
        height=480,
        channels=3,
        format="RGB",
        source="teeth.jpg",
    )
    result = DentalVisionResult(
        model="DentalTensor Vision",
        version="1.0",
        findings=[finding],
        image_metadata=metadata,
        inference_ms=42,
    )

    # Dictionary conversion
    data = result.to_dict()
    assert data["model"] == "DentalTensor Vision"
    assert data["version"] == "1.0"
    assert len(data["findings"]) == 1
    assert data["findings"][0]["class_name"] == "caries"
    assert data["findings"][0]["confidence"] == 0.71
    assert data["findings"][0]["bbox"] == (120.0, 80.0, 240.0, 190.0)
    assert data["findings"][0]["threshold_used"] == 0.55
    assert data["inference_ms"] == 42

    # JSON serialization
    json_str = result.to_json()
    parsed = json.loads(json_str)
    assert parsed["model"] == "DentalTensor Vision"
    assert parsed["version"] == "1.0"
    assert parsed["findings"][0]["class_name"] == "caries"


def test_schema_deserialization():
    """Verify DentalVisionResult can be reconstructed from raw JSON."""
    raw_payload = """
    {
      "model": "DentalTensor Vision",
      "version": "1.0",
      "findings": [
        {
          "class_name": "calculus",
          "confidence": 0.82,
          "bbox": [10.0, 20.0, 30.0, 40.0],
          "threshold_used": 0.35
        }
      ],
      "inference_ms": 50
    }
    """
    res = DentalVisionResult.model_validate_json(raw_payload)
    assert res.model == "DentalTensor Vision"
    assert len(res.findings) == 1
    assert res.findings[0].class_name == "calculus"
    assert res.findings[0].confidence == 0.82
    assert res.findings[0].threshold_used == 0.35
