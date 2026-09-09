"""DentalTensor Vision Configuration, Constants, and Model Path Resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Umbrella AI model brand & line
BRAND_NAME: str = "DentalTensor"
MODEL_FAMILY: str = "DentalTensor Vision"
VERSION: str = "1.0"
FULL_MODEL_NAME: str = "DentalTensor Vision v1.0"
DEVELOPER: str = "Nathan Asif"
BASE_ARCHITECTURE: str = "Ultralytics YOLO11n"

# Canonical 5 Finding Classes
CLASSES: Tuple[str, ...] = (
    "calculus",
    "caries",
    "gingivitis",
    "tooth_discoloration",
    "oral_ulcer",
)

# Canonical production confidence thresholds (V1 calibrated)
PRODUCTION_THRESHOLDS: Dict[str, float] = {
    "calculus": 0.35,
    "caries": 0.55,
    "gingivitis": 0.50,
    "tooth_discoloration": 0.65,
    "oral_ulcer": 0.65,
}

GLOBAL_FALLBACK_THRESHOLD: float = 0.50

# Verified benchmark metrics (Engineering benchmark on 10,698 images)
BENCHMARK_METRICS: Dict[str, float] = {
    "precision": 0.680,
    "recall": 0.666,
    "map50": 0.687,
    "map50_95": 0.360,
    "training_images": 10698,
}

CANONICAL_CHECKPOINT_NAME: str = "dentaltensor_vision_v1.0.pt"


def resolve_model_path(model_path: Optional[str | Path] = None) -> Path:
    """Resolve the path to the DentalTensor Vision pre-trained weights checkpoint.

    Resolution precedence:
    1. Explicit model_path argument if provided.
    2. DENTALTENSOR_MODEL environment variable if defined.
    3. Relative to current working directory: `models/dentaltensor_vision_v1.0.pt` or `dentaltensor_vision_v1.0.pt`.
    4. Relative to repository/package root: `<repo_root>/models/dentaltensor_vision_v1.0.pt`.
    5. Fallback relative path.
    """
    if model_path is not None:
        p = Path(model_path)
        return p

    env_path = os.getenv("DENTALTENSOR_MODEL")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    # Candidate locations
    candidates: List[Path] = [
        Path.cwd() / "models" / CANONICAL_CHECKPOINT_NAME,
        Path.cwd() / CANONICAL_CHECKPOINT_NAME,
        Path(__file__).resolve().parents[2] / "models" / CANONICAL_CHECKPOINT_NAME,
        Path(__file__).resolve().parent / "models" / CANONICAL_CHECKPOINT_NAME,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return Path("models") / CANONICAL_CHECKPOINT_NAME


def resolve_device(device: Optional[str] = "auto") -> str:
    """Resolve inference compute device ('auto', 'cpu', 'cuda').

    - 'auto': Use CUDA acceleration if PyTorch detects an available GPU; otherwise fall back to CPU.
    - 'cpu': Force standard CPU inference.
    - 'cuda' or 'cuda:X': Use CUDA device (raises RuntimeError if CUDA is unavailable).
    """
    requested = (device or "auto").strip().lower()

    if requested in ("auto", ""):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except Exception:
            return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device '{device}' was requested, but CUDA is not available on this system. "
                    f"Please run with device='cpu' or device='auto'."
                )
            return requested
        except ImportError:
            raise RuntimeError(
                f"PyTorch with CUDA support is required for device '{device}'. "
                f"Please run with device='cpu' or install a CUDA-enabled PyTorch build."
            )

    return requested


@dataclass
class DentalTensorConfig:
    """Runtime configuration for DentalTensor Vision."""

    device: str = field(
        default_factory=lambda: os.getenv("DENTALTENSOR_DEVICE", "auto").lower()
    )
    model_path: str = field(
        default_factory=lambda: os.getenv(
            "DENTALTENSOR_MODEL", "models/dentaltensor_vision_v1.0.pt"
        )
    )
    output_format: str = field(
        default_factory=lambda: os.getenv("DENTALTENSOR_OUTPUT_FORMAT", "json").lower()
    )

    def get_threshold(self, class_name: str, override: float | None = None) -> float:
        """Resolve confidence threshold for a specific class or global override."""
        if override is not None:
            return float(override)
        normalized = class_name.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in ("ulcer", "oral_ulcers"):
            normalized = "oral_ulcer"
        elif normalized in ("discoloration", "tooth_discolouration"):
            normalized = "tooth_discoloration"
        return PRODUCTION_THRESHOLDS.get(normalized, GLOBAL_FALLBACK_THRESHOLD)


# Global default configuration instance
default_config = DentalTensorConfig()
