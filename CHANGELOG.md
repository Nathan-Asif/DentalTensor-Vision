# Changelog

All notable changes to **DentalTensor Vision** are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-09-08

### Initial Public Standalone Release

#### Added
- **Core Vision Architecture**: Standalone Python package `dentaltensor-vision` (`dentaltensor` namespace).
- **Canonical Model Checkpoint**: `models/dentaltensor_vision_v1.0.pt` (SHA-256 verified, pre-trained on Ultralytics YOLO11n).
- **Pre-Trained Local Inference**: Direct offline execution on CPU and compatible local acceleration. Zero cloud accounts, zero API keys, and no training dataset required at runtime.
- **Calibrated Production Thresholds**:
  - `calculus`: 0.35
  - `caries`: 0.55
  - `gingivitis`: 0.50
  - `tooth_discoloration`: 0.65
  - `oral_ulcer`: 0.65
  - Global fallback: 0.50
- **Zero-Distortion Preprocessing**: Regression protection ensuring original RGB pixels are preserved without CLAHE or lossy compression passes.
- **Python Public API**: `DentalTensorVision()` class with `.predict()` method returning structured Pydantic `DentalVisionResult`.
- **Command-Line Interface**: `dentaltensor` CLI with `predict`, `info`, and `version` subcommands, including `--device` and `--output` options.
- **Offline Test Suite**: 36 unit and regression tests covering schemas, preprocessing, postprocessing, thresholds, CLI, device resolution, and local inference pipelines.
- **Comprehensive Documentation**: Complete `MODEL_CARD.md`, `ROADMAP.md`, `NOTICE.md`, and technical architecture docs.
