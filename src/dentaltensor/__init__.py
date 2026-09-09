"""DentalTensor — Pre-trained Oral Pathology Computer Vision.

Canonical technical positioning:
"DentalTensor Vision v1.0 is a custom-trained and fine-tuned oral pathology
computer-vision model built on the Ultralytics YOLO11n architecture and
distributed as pre-trained weights ready for inference."

Developed by Nathan Asif.
First production integration: DaantShaant.
"""

from dentaltensor.config import (
    BRAND_NAME,
    CLASSES,
    DEVELOPER,
    FIRST_INTEGRATION,
    FULL_MODEL_NAME,
    MODEL_FAMILY,
    PRODUCTION_THRESHOLDS,
    VERSION,
)
from dentaltensor.schemas import DentalFinding, DentalVisionResult, ImageMetadata
from dentaltensor.vision import DentalTensorVision

__version__ = "1.0.0"

__all__ = [
    "DentalTensorVision",
    "DentalVisionResult",
    "DentalFinding",
    "ImageMetadata",
    "CLASSES",
    "PRODUCTION_THRESHOLDS",
    "VERSION",
    "BRAND_NAME",
    "MODEL_FAMILY",
    "FULL_MODEL_NAME",
    "DEVELOPER",
    "FIRST_INTEGRATION",
    "__version__",
]
