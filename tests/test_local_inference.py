"""Test local pre-trained inference initialization, device selection, and prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from dentaltensor.config import CANONICAL_CHECKPOINT_NAME
from dentaltensor.inference.detector import DentalTensorDetector, DentalTensorInferenceError
from dentaltensor.inference.preprocessing import ImagePreprocessingError
from dentaltensor.schemas import DentalVisionResult
from dentaltensor.vision import DentalTensorVision


def test_vision_initialization_defaults():
    """Verify DentalTensorVision initializes locally with default device and canonical model."""
    vision = DentalTensorVision()
    assert vision.version == "1.0"
    assert len(vision.classes) == 5
    assert vision.device in ("cpu", "cuda")
    assert vision.model_path.name == CANONICAL_CHECKPOINT_NAME
    assert repr(vision).startswith("<DentalTensorVision")


def test_vision_initialization_cpu_device():
    """Verify DentalTensorVision initializes explicitly with device='cpu'."""
    vision = DentalTensorVision(device="cpu")
    assert vision.device == "cpu"
    assert vision._detector.device == "cpu"


def test_vision_initialization_custom_model_path():
    """Verify custom model path is assigned to detector."""
    custom_path = Path("custom_dentaltensor.pt")
    vision = DentalTensorVision(model_path=custom_path, device="cpu")
    assert vision.model_path == custom_path
    assert vision._detector.model_path == custom_path


def test_detector_missing_checkpoint_raises_filenotfound():
    """Verify FileNotFoundError is raised when checkpoint does not exist."""
    detector = DentalTensorDetector(model_path="non_existent_weights_xyz.pt")
    with pytest.raises(FileNotFoundError) as excinfo:
        detector.load_model()
    assert "model checkpoint not found" in str(excinfo.value)


def test_detector_missing_image_input_raises_error():
    """Verify error when image input is invalid or non-existent file."""
    detector = DentalTensorDetector(device="cpu")
    with pytest.raises((FileNotFoundError, ImagePreprocessingError)):
        detector.detect("non_existent_image_path_xyz.jpg")


def test_local_prediction_with_mocked_yolo_forward_pass(tmp_path):
    """Verify end-to-end local prediction pipeline using mocked YOLO outputs."""
    # Create a small blank image
    img_path = tmp_path / "test_tooth.jpg"
    img = Image.new("RGB", (640, 480), color=(200, 200, 200))
    img.save(img_path)

    vision = DentalTensorVision(device="cpu")

    # Mock Ultralytics YOLO output
    class MockBox:
        def __init__(self, xyxy, conf, cls):
            self._xyxy = xyxy
            self._conf = conf
            self._cls = cls

        def __len__(self):
            return len(self._cls)

        @property
        def xyxy(self):
            class CpuWrapper:
                def __init__(self, val):
                    self.val = val
                def cpu(self):
                    return self
                def numpy(self):
                    return self.val
            return CpuWrapper(self._xyxy)

        @property
        def conf(self):
            class CpuWrapper:
                def __init__(self, val):
                    self.val = val
                def cpu(self):
                    return self
                def numpy(self):
                    return self.val
            return CpuWrapper(self._conf)

        @property
        def cls(self):
            class CpuWrapper:
                def __init__(self, val):
                    self.val = val
                def cpu(self):
                    return self
                def numpy(self):
                    return self.val
            return CpuWrapper(self._cls)

    class MockYOLOResult:
        def __init__(self):
            # 2 detections: caries (conf 0.72) and calculus (conf 0.60)
            self.boxes = MockBox(
                xyxy=np.array([[100.0, 150.0, 200.0, 250.0], [300.0, 320.0, 400.0, 420.0]]),
                conf=np.array([0.72, 0.60]),
                cls=np.array([1.0, 0.0]),  # 1: caries, 0: calculus
            )
            self.names = {0: "calculus", 1: "caries"}

    mock_yolo = mock.MagicMock()
    mock_yolo.predict.return_value = [MockYOLOResult()]

    with mock.patch.object(vision._detector, "load_model", return_value=mock_yolo):
        result = vision.predict(str(img_path))

    assert isinstance(result, DentalVisionResult)
    assert result.model == "DentalTensor Vision"
    assert result.version == "1.0"
    assert len(result.findings) == 2

    caries_finding = next(f for f in result.findings if f.class_name == "caries")
    assert caries_finding.confidence == 0.72
    assert caries_finding.threshold_used == 0.55
    assert caries_finding.bbox == (100.0, 150.0, 200.0, 250.0)

    calculus_finding = next(f for f in result.findings if f.class_name == "calculus")
    assert calculus_finding.confidence == 0.60
    assert calculus_finding.threshold_used == 0.35
    assert calculus_finding.bbox == (300.0, 320.0, 400.0, 420.0)

    assert result.image_metadata is not None
    assert result.image_metadata.width == 640
    assert result.image_metadata.height == 480
