"""Test configuration, model path resolution, and device selection."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from dentaltensor.config import (
    CANONICAL_CHECKPOINT_NAME,
    DentalTensorConfig,
    GLOBAL_FALLBACK_THRESHOLD,
    PRODUCTION_THRESHOLDS,
    resolve_device,
    resolve_model_path,
)


def test_default_config():
    """Verify default config values for local inference."""
    cfg = DentalTensorConfig()
    assert cfg.device == "auto"
    assert "dentaltensor_vision_v1.0.pt" in cfg.model_path
    assert cfg.output_format == "json"


def test_config_env_overrides():
    """Verify environment variables override configuration."""
    with mock.patch.dict(
        os.environ,
        {
            "DENTALTENSOR_DEVICE": "cpu",
            "DENTALTENSOR_OUTPUT_FORMAT": "json",
            "DENTALTENSOR_MODEL": "custom/path/model.pt",
        },
    ):
        cfg = DentalTensorConfig()
        assert cfg.device == "cpu"
        assert cfg.model_path == "custom/path/model.pt"


def test_config_threshold_resolver():
    """Verify threshold resolution on DentalTensorConfig."""
    cfg = DentalTensorConfig()
    # Class-specific defaults
    assert cfg.get_threshold("caries") == 0.55
    assert cfg.get_threshold("calculus") == 0.35
    assert cfg.get_threshold("tooth_discoloration") == 0.65
    assert cfg.get_threshold("tooth discoloration") == 0.65
    assert cfg.get_threshold("oral_ulcer") == 0.65
    assert cfg.get_threshold("ulcer") == 0.65
    assert cfg.get_threshold("unknown_class") == GLOBAL_FALLBACK_THRESHOLD

    # Explicit override takes priority
    assert cfg.get_threshold("caries", override=0.75) == 0.75
    assert cfg.get_threshold("calculus", override=0.20) == 0.20


def test_resolve_device_cpu():
    """Verify explicit CPU device resolution."""
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("CPU") == "cpu"


def test_resolve_device_auto():
    """Verify auto device resolution defaults cleanly."""
    dev = resolve_device("auto")
    assert dev in ("cpu", "cuda")


def test_resolve_device_cuda_unavailable():
    """Verify clear error when CUDA is requested but unavailable."""
    with mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError) as excinfo:
            resolve_device("cuda")
        assert "CUDA device 'cuda' was requested" in str(excinfo.value)


def test_resolve_model_path_explicit():
    """Verify explicit model path is respected."""
    p = resolve_model_path("custom_weights.pt")
    assert p == Path("custom_weights.pt")


def test_resolve_model_path_finds_shipped_checkpoint():
    """Verify resolution finds the canonical shipped checkpoint."""
    p = resolve_model_path()
    assert p.name == CANONICAL_CHECKPOINT_NAME
