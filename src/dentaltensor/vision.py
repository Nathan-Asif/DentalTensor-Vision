"""Main user-facing Python API for DentalTensor Vision."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from dentaltensor.config import (
    BRAND_NAME,
    CLASSES,
    FULL_MODEL_NAME,
    PRODUCTION_THRESHOLDS,
    VERSION,
    default_config,
    resolve_device,
    resolve_model_path,
)
from dentaltensor.inference.detector import DentalTensorDetector
from dentaltensor.inference.preprocessing import ImageInputType
from dentaltensor.schemas import DentalVisionResult


class DentalTensorVision:
    """DentalTensor Vision v1.0 — Pre-trained Oral Pathology Vision Engine.

    Local, pre-trained computer vision model for detecting oral pathology from intraoral images.
    Built on the Ultralytics YOLO11n architecture. Runs locally without cloud dependencies.

    Example:
    >>> from dentaltensor import DentalTensorVision
    >>> vision = DentalTensorVision()
    >>> result = vision.predict("path/to/teeth.jpg")
    >>> print(result.to_json())
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        **kwargs,
    ):
        """Initialize DentalTensor Vision with pre-trained weights.

        Parameters:
        - model_path: Optional path to model checkpoint (.pt). If omitted, resolves
                      the canonical distributed checkpoint `models/dentaltensor_vision_v1.0.pt`.
        - device: Compute device ('auto', 'cpu', 'cuda'). Defaults to 'auto' (detects
                  compatible acceleration, otherwise falls back to standard CPU).
        """
        self.model_name = FULL_MODEL_NAME
        self.version = VERSION
        self.classes = CLASSES
        self.thresholds = dict(PRODUCTION_THRESHOLDS)
        self.device = resolve_device(device)
        self.model_path = resolve_model_path(model_path)

        self._detector = DentalTensorDetector(
            model_path=self.model_path,
            device=self.device,
        )

    def predict(
        self,
        image: ImageInputType,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs,
    ) -> DentalVisionResult:
        """Run oral pathology detection on an image.

        Parameters:
        - image: Path to image file, raw bytes, PIL Image, or numpy array.
        - threshold: Optional global confidence threshold override. If omitted,
                     the calibrated class-specific production thresholds are applied.
        - device: Optional device override for this prediction ('cpu', 'cuda', 'auto').
        - source: Optional source identifier (filename). If omitted and image is a path,
                  the filename is automatically used.

        Returns:
        - DentalVisionResult: Structured result containing findings and metadata.
        """
        source_id = Path(source).name if source is not None else None
        return self._detector.detect(
            image_input=image,
            threshold_override=threshold,
            device=device,
            source_id=source_id,
            **kwargs,
        )

    def get_thresholds(self) -> dict[str, float]:
        """Return the active class-specific confidence thresholds."""
        return dict(self.thresholds)

    def __repr__(self) -> str:
        return (
            f"<DentalTensorVision version='{self.version}' "
            f"device='{self.device}' classes={len(self.classes)}>"
        )
