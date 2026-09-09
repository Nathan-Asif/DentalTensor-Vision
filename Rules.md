# DentalTensor Vision — Agent Instructions

You are working on the standalone open-source DentalTensor Vision repository.

## MANDATORY WORKFLOW

Before making ANY edit:

1. READ `/context.md` completely.
2. Run:
   - `git status`
   - `git diff`
3. Inspect the existing repository before changing anything.
4. Preserve all existing/manual work. Current code is the source of truth.
5. Do not perform broad refactors unless explicitly requested.

## GIT SAFETY

NEVER run:

- git add
- git commit
- git push
- git reset --hard
- git clean -fd
- destructive history operations

Nathan handles all Git mutations manually.
You may run read-only Git commands.

## EXECUTION SAFETY

Do NOT:

- open a browser
- launch localhost for manual testing
- run Playwright
- run Selenium
- perform live external network or cloud calls
- deploy to Modal or any cloud provider
- use real external credentials

You MAY run:

- unit tests
- mocked tests
- type/static checks
- package builds
- CLI tests
- offline model metadata checks
- local CPU/accelerated inference

Nathan performs live acceptance testing manually.

## SECRETS

Never include or print:

- API keys
- tokens
- production secrets
- DaantShaant secrets

DentalTensor Vision v1.0 requires no credentials or API keys.

## PRODUCT IDENTITY

Umbrella brand:
DentalTensor

Model family:
DentalTensor Vision

Current release:
DentalTensor Vision v1.0

Developer:
Nathan Asif

First production integration:
DaantShaant

Repository:
DentalTensor-Vision

Python distribution:
dentaltensor-vision

Python namespace:
dentaltensor

CLI:
dentaltensor

## TECHNICAL CLAIM

Use:

"DentalTensor Vision v1.0 is a custom-trained and fine-tuned oral
pathology computer-vision model built on the Ultralytics YOLO11n
architecture and distributed as pre-trained weights ready for inference."

Do NOT claim:

- YOLO was created by DentalTensor
- model architecture was created from scratch
- users need to train the model
- users need to download the 10,698-image dataset
- definitive clinical diagnosis
- guaranteed medical accuracy

DentalTensor engineering includes custom dental training, dataset work,
hard-negative mining, calibration, class-specific thresholds,
inference engineering, and production validation.

## MEDICAL LANGUAGE

Use:

- visual screening
- visual finding
- possible concern
- detection
- screening model

Avoid definitive medical diagnosis claims.

## LICENSE

Repository license:
GNU Affero General Public License v3.0

SPDX:
AGPL-3.0-only

Preserve required attribution to Ultralytics YOLO.

## V1 SCOPE — PRE-TRAINED LOCAL INFERENCE

DentalTensor Vision v1.0 is PRE-TRAINED.
V1 is inference-only for users.
Users do not train it.
Users do not need the dataset.
Users do not need Modal or any cloud account.
Users do not need API keys.
The shipped `.pt` file (`models/dentaltensor_vision_v1.0.pt`) contains the learned model parameters.
CPU inference is supported by default.
Compatible acceleration may improve performance.
Modal/cloud deployment is NOT part of V1 runtime.

Pipeline:
Image
→ DentalTensor Vision (pre-trained YOLO11n)
→ calibrated thresholding
→ structured visual findings

No:
- Qwen
- Gemini
- OpenAI
- Claude
- LangGraph
- Supabase
- DaantShaant runtime dependency
- Modal runtime dependency

## FUTURE — DO NOT IMPLEMENT NOW

V2:
Advanced inference and deployment options (CUDA optimizations, batching, ONNX, TensorRT, optional hosted inference).

V3:
Optional pluggable LLM/report adapters (Qwen, Gemini, Claude, OpenAI, local LLMs).

These belong only in ROADMAP.md.

## FINAL RESPONSE FORMAT

Every implementation task must finish with:

1. Files created/changed
2. What was implemented
3. Tests/checks executed
4. Results
5. Any unresolved issue
6. `git status`
7. Exact manual commands Nathan should run next