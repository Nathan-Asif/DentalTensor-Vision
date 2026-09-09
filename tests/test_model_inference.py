"""Integration tests using the real pre-trained model checkpoint without mocking YOLO."""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

from dentaltensor.config import CANONICAL_CHECKPOINT_NAME, PRODUCTION_THRESHOLDS
from dentaltensor.vision import DentalTensorVision

EXPECTED_SHA256 = "42bf517ded4eb15eebe6b5361ebf9e6ab21d8488098c4e3e4ccfe5912ecbbe27"
REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = REPO_ROOT / "models" / CANONICAL_CHECKPOINT_NAME
YELLOWISH_IMAGE = REPO_ROOT / "Test Images" / "Yellowish-Teeth.jpg"
BAD_TEETH_IMAGE = REPO_ROOT / "Test Images" / "bad_teeth.jpeg"


@pytest.mark.model
def test_checkpoint_integrity():
    """Verify pre-trained weights file exists and matches the byte-for-byte SHA-256."""
    assert CHECKPOINT_PATH.is_file(), f"Model checkpoint not found at {CHECKPOINT_PATH}"
    h = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    assert h == EXPECTED_SHA256, f"Checkpoint hash mismatch: got {h}, expected {EXPECTED_SHA256}"


@pytest.mark.model
def test_real_model_classes_and_names():
    """Verify raw YOLO model names match expected source-of-truth classes."""
    pytest.importorskip("ultralytics")
    vision = DentalTensorVision(model_path=CHECKPOINT_PATH, device="cpu")
    yolo_model = vision._detector.load_model()
    names = getattr(yolo_model, "names", {})

    # Print model.names once during tests as requested
    print(f"\n[test_real_model_classes_and_names] Raw model.names: {names}")

    expected_raw = {
        0: "calculus",
        1: "caries",
        2: "gingivitis",
        3: "tooth discoloration",
        4: "ulcer",
    }
    for idx, expected_name in expected_raw.items():
        assert names.get(idx) == expected_name, f"Index {idx} expected {expected_name}, got {names.get(idx)}"


@pytest.mark.model
def test_real_inference_yellowish_teeth():
    """Verify real inference on Yellowish-Teeth.jpg produces tooth_discoloration detections."""
    if not YELLOWISH_IMAGE.is_file():
        pytest.skip(f"Test image not found: {YELLOWISH_IMAGE}")

    vision = DentalTensorVision(model_path=CHECKPOINT_PATH, device="cpu")

    # Diagnostic override
    diag_result = vision.predict(YELLOWISH_IMAGE, threshold=0.01)
    assert diag_result.model == "DentalTensor Vision"
    assert diag_result.image_metadata is not None
    assert diag_result.image_metadata.source == "Yellowish-Teeth.jpg"
    assert len(diag_result.findings) > 0

    discoloration_findings = [f for f in diag_result.findings if f.class_name == "tooth_discoloration"]
    assert len(discoloration_findings) > 0, "Expected tooth_discoloration findings on Yellowish-Teeth.jpg"

    # Verify highest confidence matches the known production score (~0.75)
    max_conf = max(f.confidence for f in discoloration_findings)
    assert max_conf >= 0.70, f"Expected highest tooth_discoloration confidence >= 0.70, got {max_conf}"

    # Production threshold inference
    prod_result = vision.predict(YELLOWISH_IMAGE)
    assert prod_result.image_metadata.source == "Yellowish-Teeth.jpg"
    # Detections >= 0.65 threshold should be present
    prod_discoloration = [f for f in prod_result.findings if f.class_name == "tooth_discoloration"]
    assert len(prod_discoloration) >= 5, f"Expected multiple findings above 0.65 threshold, got {len(prod_discoloration)}"
    for f in prod_discoloration:
        assert f.confidence >= PRODUCTION_THRESHOLDS["tooth_discoloration"]
        assert f.threshold_used == PRODUCTION_THRESHOLDS["tooth_discoloration"]


@pytest.mark.model
def test_real_inference_bad_teeth_two_stage_filtering():
    """Verify real inference on bad_teeth.jpeg filters borderline caries (<0.55) in production."""
    if not BAD_TEETH_IMAGE.is_file():
        pytest.skip(f"Test image not found: {BAD_TEETH_IMAGE}")

    vision = DentalTensorVision(model_path=CHECKPOINT_PATH, device="cpu")

    # Diagnostic mode
    diag_result = vision.predict(BAD_TEETH_IMAGE, threshold=0.01)
    assert diag_result.image_metadata.source == "bad_teeth.jpeg"
    caries_diag = [f for f in diag_result.findings if f.class_name == "caries"]
    assert len(caries_diag) >= 3

    # Production mode: only caries >= 0.55 kept
    prod_result = vision.predict(BAD_TEETH_IMAGE)
    assert prod_result.image_metadata.source == "bad_teeth.jpeg"
    for f in prod_result.findings:
        if f.class_name == "caries":
            assert f.confidence >= 0.55
            assert f.threshold_used == 0.55
