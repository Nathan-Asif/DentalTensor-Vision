"""Data models and schemas for DentalTensor Vision outputs."""

from __future__ import annotations

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class DentalFinding(BaseModel):
    """A single detected visual oral finding."""

    class_name: str = Field(
        ...,
        description="Canonical finding class name (calculus, caries, gingivitis, tooth_discoloration, oral_ulcer)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0.0 and 1.0",
    )
    bbox: Tuple[float, float, float, float] = Field(
        ...,
        description="Bounding box coordinates in pixels: [x1, y1, x2, y2]",
    )
    threshold_used: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence threshold applied for filtering this class",
    )
    bbox_normalized: Optional[Tuple[float, float, float, float]] = Field(
        default=None,
        description="Normalized bounding box coordinates [x1_norm, y1_norm, x2_norm, y2_norm] (0.0 to 1.0)",
    )


class ImageMetadata(BaseModel):
    """Metadata describing the input image processed by the vision model."""

    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    channels: int = Field(default=3, description="Number of color channels")
    format: str = Field(default="RGB", description="Color format (e.g. RGB, BGR)")
    source: Optional[str] = Field(
        default=None, description="Source file path, URL, or identifier"
    )


class DentalVisionResult(BaseModel):
    """Structured result returned by DentalTensor Vision inference."""

    model: str = Field(
        default="DentalTensor Vision", description="Umbrella vision model name"
    )
    version: str = Field(default="1.0", description="Model release version")
    findings: List[DentalFinding] = Field(
        default_factory=list,
        description="List of detected visual findings meeting confidence thresholds",
    )
    image_metadata: Optional[ImageMetadata] = Field(
        default=None, description="Dimensions and format of processed image"
    )
    inference_ms: Optional[int] = Field(
        default=None, description="Inference execution duration in milliseconds"
    )

    def to_dict(self) -> dict:
        """Serialize result to Python dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize result to formatted JSON string."""
        return self.model_dump_json(indent=indent)
