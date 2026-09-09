# DentalTensor

## DentalTensor Vision v1.0

**Pre-trained open-source dental computer vision.**  
Pass an oral image. Receive structured visual findings.  
No training required.

Developed by **Nathan Asif**.  
First production integration: **DaantShaant**.

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-orange.svg)](LICENSE)
[![Base Architecture: YOLO11n](https://img.shields.io/badge/Architecture-Ultralytics%20YOLO11n-green.svg)](https://github.com/ultralytics/ultralytics)
[![Inference: Local](https://img.shields.io/badge/Inference-Local%20%28CPU%20%2F%20Acceleration%29-brightgreen.svg)](#how-inference-works)
[![Release](https://img.shields.io/badge/Version-v1.0-brightgreen.svg)](https://github.com/Nathan-Asif/DentalTensor-Vision/releases)

---

### Key Verified Metrics at a Glance

| 📊 Training Images | 🏷️ Classes | 🎯 mAP@50 | ⚖️ Precision | 🔍 Recall |
| :---: | :---: | :---: | :---: | :---: |
| **10,698** | **5** | **0.687** | **0.680** | **0.666** |

> [!IMPORTANT]
> **THE MODEL IS ALREADY TRAINED.**  
> DentalTensor Vision v1.0 ships with pre-trained learned weights (`models/dentaltensor_vision_v1.0.pt`).  
> **Users do not need to download the training dataset or configure training infrastructure.**  
> No training is required. No cloud account is required. No API keys are required.  
> **Install. Load. Predict.**

---

## 1. DentalTensor Vision

**DentalTensor Vision v1.0** is a custom-trained and fine-tuned oral pathology computer-vision model built on the Ultralytics YOLO11n architecture and distributed as pre-trained weights ready for inference.

Conceived, trained, and productized by **Nathan Asif**, DentalTensor provides a dedicated, lightweight, high-speed vision detector engineered specifically for intraoral photographic conditions.

DentalTensor-specific engineering includes:
- **Dental dataset preparation** across 10,698 dental images
- **Task-specific fine-tuning** on YOLO11n architecture
- **Hard-negative mining** with healthy dentition pools to suppress false positives
- **Empirical confidence calibration** tailored to intraoral photography
- **Class-specific production thresholds** to balance sensitivity and specificity
- **Destructive preprocessing regression prevention** (no CLAHE / no lossy compression)
- **Production validation** in the DaantShaant oral health screening platform

---

## 2. What It Does

DentalTensor Vision detects visible oral findings directly from standard 2D photographs (JPEG, PNG, WebP) of human teeth and oral mucosa, outputting structured bounding boxes, confidence values, and calibrated class labels.

### Visual Architecture

```
                 DENTAL IMAGE (JPEG / PNG / WebP)
                                │
                                ▼
                    DENTALTENSOR VISION v1.0
                  (Pre-trained YOLO11n weights)
                                │
                                ▼
                         OBJECT DETECTION
                    (Raw bounding boxes & scores)
                                │
                                ▼
                    CLASS-SPECIFIC CALIBRATION
          (Calculus: 0.35 | Caries: 0.55 | Gingivitis: 0.50
           Tooth Discoloration: 0.65 | Oral Ulcer: 0.65)
                                │
                                ▼
                    STRUCTURED VISUAL FINDINGS
                  (Normalized BBoxes, Confidences, JSON)
```

```
[Developer Application / CLI] ──► [DentalTensor Vision] ──► [Structured JSON / Python Result]
```

---

## 3. Real Inference Example

```python
from dentaltensor import DentalTensorVision

# 1. Load pre-trained DentalTensor weights
vision = DentalTensorVision()

# 2. Pass dental image and run local inference
result = vision.predict("teeth.jpg")

# 3. Print structured visual findings
print(result.to_json(indent=2))
```

---

## 4. Quick Start

### Installation

Clone the repository and install the package:

```bash
git clone https://github.com/Nathan-Asif/DentalTensor-Vision.git
cd DentalTensor-Vision

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

# Install DentalTensor Vision
pip install -e .
```

### Run Prediction (CLI)

```bash
dentaltensor predict path/to/teeth.jpg --json
```

### Run Prediction (Python)

```python
from dentaltensor import DentalTensorVision

model = DentalTensorVision()
result = model.predict("teeth.jpg")
print(result.to_json())
```

That is all that is required to use the model.

---

## 5. Python API

DentalTensor Vision is designed as a clean, reusable AI library:

```python
from dentaltensor import DentalTensorVision

# Initialize with pre-trained weights (auto-detects local acceleration or uses CPU)
vision = DentalTensorVision()

# Run prediction
result = vision.predict("path/to/teeth.jpg")

# Access findings programmatically
for finding in result.findings:
    print(
        finding.class_name,
        finding.confidence,
        finding.bbox,
    )
```

### Device Selection

```python
# Default: auto-detects acceleration if available, falls back to CPU
vision = DentalTensorVision(device="auto")

# Force standard CPU inference (no GPU required)
vision_cpu = DentalTensorVision(device="cpu")

# Explicit CUDA acceleration (if supported)
vision_cuda = DentalTensorVision(device="cuda")
```

### Confidence Threshold Override

```python
# Override per-class cutoffs with a strict global cutoff
result = vision.predict("teeth.jpg", threshold=0.70)
```

---

## 6. CLI (Command-Line Interface)

DentalTensor Vision provides an offline command-line interface: `dentaltensor`.

```bash
# Formatted table output
dentaltensor predict teeth.jpg

# Structured JSON output
dentaltensor predict teeth.jpg --json

# Save JSON findings to file
dentaltensor predict teeth.jpg --output result.json

# Select compute device
dentaltensor predict teeth.jpg --device cpu
dentaltensor predict teeth.jpg --device cuda

# View model architecture, metrics, and thresholds
dentaltensor info

# View version
dentaltensor version
```

---

## 7. Output Schema

Structured visual findings are returned with deterministic coordinates and metadata:

```json
{
  "model": "DentalTensor Vision",
  "version": "1.0",
  "findings": [
    {
      "class_name": "caries",
      "confidence": 0.71,
      "bbox": [145.2, 210.5, 230.8, 295.1],
      "threshold_used": 0.55,
      "bbox_normalized": [0.2269, 0.4385, 0.3606, 0.6148]
    },
    {
      "class_name": "calculus",
      "confidence": 0.64,
      "bbox": [280.0, 310.2, 340.5, 365.0],
      "threshold_used": 0.35,
      "bbox_normalized": [0.4375, 0.6463, 0.5320, 0.7604]
    }
  ],
  "image_metadata": {
    "width": 640,
    "height": 480,
    "channels": 3,
    "format": "RGB",
    "source": "teeth.jpg"
  },
  "inference_ms": 42
}
```

---

## 8. Supported Findings

| Model Class Name | Canonical Code | Visual Clinical Description | Calibrated Cutoff |
|---|---|---|---|
| `calculus` | `calculus` | Supragingival and visible subgingival dental calculus / tartar | **0.35** |
| `caries` | `caries` | Visible enamel cavitation, dark lesions, and occlusal decay suspects | **0.55** |
| `gingivitis` | `gingivitis` | Marginal gingival erythema, swelling, and localized gum inflammation | **0.50** |
| `tooth discoloration` | `tooth_discoloration` | Extrinsic stains, fluorosis staining, and enamel color variation | **0.65** |
| `oral ulcer` | `oral_ulcer` | Aphthous stomatitis, oral mucosal ulcerations, and visible sores | **0.65** |

---

## 9. Model Performance

All metrics represent audited evaluations on **1,070 untouched validation images** and **1,070 untouched test images**:

| Metric | Benchmark Score |
|---|---|
| **Precision** | **0.680** |
| **Recall** | **0.666** |
| **mAP@50** | **0.687** |
| **mAP@50-95** | **0.360** |
| **Training Set** | 10,698 dental images |
| **Total Parameters** | 2,583,127 (YOLO11n) |
| **GFLOPs** | 6.4 (at 640x640) |

### Training Diagnostics

| Training Convergence | Confusion Matrix |
| :---: | :---: |
| ![Training Results](assets/metrics/training_results.png) | ![Confusion Matrix](assets/metrics/confusion_matrix.png) |

---

## 10. How Inference Works

DentalTensor Vision v1.0 implements **local inference**. It executes completely offline without external network access after dependencies and weights are installed.

- **Pre-Trained Weights**: Ships with `models/dentaltensor_vision_v1.0.pt` (5,450,330 bytes, SHA-256: `42BF517DED4EB15EEBE6B5361EBF9E6AB21D8488098C4E3E4CCFE5912ECBBE27`).
- **CPU Inference**: Standard CPU execution is fully supported. No special GPU or CUDA hardware is required.
- **Hardware Acceleration**: If compatible local acceleration is available through PyTorch, the engine utilizes it automatically or via `device="cuda"`.
- **Zero-Distortion Pipeline**: Decodes RGB pixels directly without applying CLAHE or lossy compression passes that could suppress fine pathology boundaries.

---

## 11. Training Provenance vs. Inference

The distinction between model development and end-user inference is absolute:

```
TRAINING (Already completed by DentalTensor development):
10,698+ dental images ──► fine-tuning & calibration ──► learned parameters ──► dentaltensor_vision_v1.0.pt

INFERENCE (What users perform):
dentaltensor_vision_v1.0.pt + new dental image ──► DentalTensor Vision ──► structured visual findings
```

> [!NOTE]
> **Dataset Not Required**: The training dataset is not required for inference. DentalTensor Vision v1.0 ships with the learned model weights. Users only perform inference.

### Dataset Provenance (For Research Transparency)
- **Primary Pathology Dataset (`oral-disease.yolov11`)**: 10,698 dental images (Train: 8,558, Val: 1,070, Test: 1,070) with 62,720 annotated bounding boxes and 570 empty-label negative controls. License: CC BY 4.0.
- **Auxiliary Controls**: Healthy dentition pools mined to aggressively suppress false-positive discoloration predictions on clean teeth.

---

## 12. Calibration

Raw model predictions exhibit varying confidence distributions across condition classes. DentalTensor applies empirical, class-specific production cutoffs:

- **Calculus (0.35)**: High sensitivity for faint tartar deposits along the gingival line.
- **Caries (0.55)**: Conservative threshold preventing false alarms on deep occlusal fissures.
- **Gingivitis (0.50)**: Balanced threshold for visible gum erythema.
- **Tooth Discoloration (0.65)**: Elevated cutoff preventing natural tooth shade variation from triggering alerts.
- **Oral Ulcer (0.65)**: Specific threshold for distinct mucosal ulcerations.

---

## 13. Limitations & Medical Disclaimer

> [!CAUTION]
> **Not a Medical Device**: DentalTensor Vision v1.0 is an artificial intelligence research and visual screening tool. It is **not** a certified medical device, does not provide definitive clinical diagnoses, and must never replace an examination by a licensed dentist or oral healthcare professional.

- **Subgingival & Interproximal Pathology**: Standard RGB photography cannot detect subgingival calculus concealed below the gumline or interproximal decay between tight contacts. Dental radiographs (bitewing/periapical X-rays) remain mandatory.
- **Visual Variations**: Benign pigmentation, restorative margins, and natural dental anatomy can occasionally resemble pathology under poor lighting or severe glare.
- **Image Quality**: Motion blur, extreme underexposure, or low resolution can degrade detection confidence.

---

## 14. First Production Integration: DaantShaant

DentalTensor was conceived, trained, and battle-tested within **DaantShaant**, an oral health screening and care-navigation platform created by Nathan Asif for the Alibaba Cloud Bano Qabil Hackathon 2026. Within DaantShaant, DentalTensor Vision serves as the foundational perception layer powering automated dental assessments.

---

## 15. Roadmap

### v1.x (Current)
Pre-trained local inference:
- CPU inference
- Compatible local acceleration
- Python API (`DentalTensorVision()`)
- CLI (`dentaltensor predict`)
- Calibrated production thresholds
- Model quality patches and evaluation tools
- Packaging improvements

### v2.0
Advanced inference and deployment options:
- Explicit optimized CUDA workflows
- Batching pipelines
- ONNX export
- TensorRT acceleration
- Additional runtime backends
- Optional hosted inference
- Improved performance and model variants

### v3.0
Optional pluggable AI/report layer:
```
DentalTensor Vision ──► structured findings ──► optional AI provider ──► formatted report / downstream workflow
```
Potential provider adapters: Qwen, Gemini, OpenAI, Claude, local LLM, custom provider.  
*The core vision model will always remain independently usable without an LLM.*

---

## 16. License

DentalTensor Vision is released under the **GNU Affero General Public License v3.0 (AGPL-3.0-only)**.  
See the full text in [LICENSE](LICENSE).

### Attribution
DentalTensor Vision v1.0 is built on the Ultralytics YOLO11n architecture. Full third-party notices and acknowledgments are documented in [NOTICE.md](NOTICE.md).

---

## 17. Citation

If you use DentalTensor Vision in your research or application, please cite:

```bibtex
@software{dentaltensor_vision_2026,
  author = {Nathan Asif},
  title = {DentalTensor Vision: Open-Source Oral Pathology Computer Vision},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/Nathan-Asif/DentalTensor-Vision}
}
```

---

## 18. Contributing

Contributions are welcomed! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, testing, and pull requests.
