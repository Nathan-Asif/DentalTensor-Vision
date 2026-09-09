# DentalTensor Vision — Context & Source of Truth

## Product Identity & Branding Hierarchy
- **Umbrella AI Model Brand:** DentalTensor
- **Model Line:** DentalTensor Vision
- **Public Release:** DentalTensor Vision v1.0
- **Developer:** Nathan Asif
- **First Production Integration:** DaantShaant
- **Repository:** DentalTensor-Vision
- **Package Distribution Name:** dentaltensor-vision
- **Import Namespace:** dentaltensor
- **CLI Command:** dentaltensor

## Canonical Technical Positioning
"DentalTensor Vision v1.0 is a custom-trained and fine-tuned oral pathology computer-vision model built on the Ultralytics YOLO11n architecture and distributed as pre-trained weights ready for inference."

DentalTensor-specific engineering includes:
- Dental dataset preparation (10,698 dental images)
- Task-specific training & fine-tuning
- Hard-negative mining and healthy-pool mining
- Confidence calibration
- Class-specific production thresholds
- Inference engineering
- Production validation inside DaantShaant

Do NOT claim the underlying YOLO architecture was created from scratch.
Do NOT describe DentalTensor as merely a renamed YOLO model.
Do NOT say users have to train DentalTensor.
Do NOT imply the 10,698-image dataset must be downloaded for runtime.

## V1 Scope & Architecture
- **Pre-Trained Model Product**: The model has ALREADY been trained.
- **Inference Only**: Users only perform inference.
- **Learned Parameters**: Shipped canonical weights `models/dentaltensor_vision_v1.0.pt`.
- **Local Inference**: Executes locally on CPU or local compatible acceleration.
- **No Cloud Services**: Zero Modal accounts, zero cloud dependencies, zero external API keys.
- **No Dataset at Runtime**: The training dataset is NOT required for inference.
- Standalone vision only.
- Strict NO dependencies on: DaantShaant, Qwen, Gemini, OpenAI, Claude, LangGraph, Supabase.
- No natural-language clinical report generation in V1.
- No definitive medical diagnosis claims. Use terms like visual screening, visual finding, detection, possible concern.

## Model Facts
- **Canonical Model Name:** DentalTensor Vision v1.0
- **Public Repository Checkpoint Destination:** `models/dentaltensor_vision_v1.0.pt`
- **Byte Size:** 5,450,330 bytes
- **SHA-256:** `42BF517DED4EB15EEBE6B5361EBF9E6AB21D8488098C4E3E4CCFE5912ECBBE27`
- **Primary Training Dataset:** 10,698 dental images (historical provenance only, not runtime)
- **Classes (5 classes):**
  - `0`: `calculus` (calculus / tartar)
  - `1`: `caries` (caries / possible decay)
  - `2`: `gingivitis` (gingivitis signs)
  - `3`: `tooth_discoloration` (tooth discoloration)
  - `4`: `oral_ulcer` (oral ulcer)

## Verified Validation Metrics
- Precision: 0.680
- Recall: 0.666
- mAP@50: 0.687
- mAP@50-95: 0.360
(Do NOT invent stronger metrics).

## Production Confidence Thresholds
Centralized thresholds:
- Global fallback: 0.50
- `calculus`: 0.35
- `caries`: 0.55
- `gingivitis`: 0.50
- `tooth_discoloration`: 0.65
- `oral_ulcer`: 0.65

## Important Preprocessing Regression
During DaantShaant development, extra CLAHE and aggressive JPEG re-encoding were found to suppress real YOLO detections.
The known-good DentalTensor inference path uses the decoded image without unnecessary destructive preprocessing.
- DO NOT add CLAHE to default inference.
- DO NOT JPEG-reencode images before YOLO unnecessarily.
- Preserve original decoded image quality.
- Include a regression unit test protecting this behavior.

## Compute & Device Behavior
- **Default Device:** `auto` (uses compatible local acceleration if available, else falls back to CPU).
- **CPU Inference:** Standard CPU execution is supported. No GPU is required.
- **Explicit Devices:** `cpu`, `cuda`.
- Modal/cloud deployment is NOT part of V1 runtime.

## Architecture & Roadmap Boundaries
- **V1 (Current):** Developer app / CLI -> Python API (`dentaltensor`) -> Local Pre-trained Inference (CPU/acceleration) -> Structured JSON findings.
- **V2 (Future — Roadmap only):** Advanced inference and deployment options (CUDA optimizations, batching, ONNX, TensorRT, optional hosted inference).
- **V3 (Future — Roadmap only):** Optional provider-neutral AI/report generation adapters (Qwen, Gemini, OpenAI, Claude, local LLM). DentalTensor Vision remains usable without LLMs.

## Licensing & Attribution
- License: GNU Affero General Public License v3.0 (`AGPL-3.0-only`).
- Preserve required attribution to Ultralytics YOLO.

## Git Safety & Agent Operating Rules
- NEVER run `git add`, `git commit`, `git push`, `git reset --hard`, `git clean -fd`.
- Nathan handles all Git mutations manually.
- No live browser, no external network calls during agent execution.
