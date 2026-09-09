# Model Card: DentalTensor Vision v1.0

Developed by **Nathan Asif**  
First Production Integration: **DaantShaant**

---

## 1. Model Details

- **Model Name**: DentalTensor Vision
- **Version**: 1.0.0 (`v1.0`)
- **Developer & Author**: Nathan Asif
- **Model Type**: Object detection network (bounding-box visual finding detector)
- **Base Architecture**: Ultralytics YOLO11n (nano object detector, 101 layers, 2,583,127 parameters, 6.4 GFLOPs)
- **Release Date**: 2026
- **License**: GNU Affero General Public License v3.0 (`AGPL-3.0-only`)
- **Primary Checkpoint**: `models/dentaltensor_vision_v1.0.pt` (SHA-256: `42BF517DED4EB15EEBE6B5361EBF9E6AB21D8488098C4E3E4CCFE5912ECBBE27`, 5,450,330 bytes)
- **Input Resolution**: 640x640 RGB image (`imgsz=640`)
- **Compute Runtime**: Local inference (CPU supported; local compatible acceleration optional)

---

## 2. Training Provenance vs. Inference Requirements

```
TRAINING PROVENANCE (Development Phase — Completed)
10,698 dental images ──► fine-tuning & calibration ──► learned parameters ──► dentaltensor_vision_v1.0.pt

INFERENCE (Runtime Phase — End User)
dentaltensor_vision_v1.0.pt + new dental image ──► DentalTensor Vision ──► structured visual findings
```

### Inference Requirements (End Users)
- **Pre-trained Checkpoint**: Shipped directly with the repository (`models/dentaltensor_vision_v1.0.pt`).
- **No Training Dataset Required**: Users do not need to download or store the 10,698-image training dataset.
- **No Retraining Required**: The model is fully fine-tuned and ready for immediate inference.
- **Offline / Local Execution**: No internet connection is required after dependencies and weights are installed.
- **No Cloud Accounts or API Keys**: Operates locally without external cloud dependencies.
- **Hardware**: Standard CPU is fully supported. Compatible acceleration (e.g. CUDA) may be used if available.

### Training Provenance (Research & Auditing Only)
The model was fine-tuned and calibrated across 10,698 annotated intraoral images using AdamW optimizer at 640x640 resolution. The training dataset is documented solely for model provenance, auditability, and academic reproducibility.

---

## 3. Intended Use & Clinical Positioning

### Intended Use Cases
- Preliminary computer-assisted screening of visible dental and oral mucosal conditions from standard 2D RGB mouth photographs.
- Spatial localization of visible enamel cavitation, tartar deposits, gingival inflammation, staining, and mucosal ulcerations.
- Educational, tele-dentistry triaging, and health awareness applications.
- Integration as a foundational vision perception asset in digital health pipelines (e.g., DaantShaant).

### Out-of-Scope & Prohibited Uses
- **Definitive Medical Diagnosis**: DentalTensor Vision is not certified for standalone clinical diagnostic determination.
- **Surgical or Treatment Planning**: The model cannot assess structural tooth vitality, bone levels, or pulp involvement.
- **Replacement for Diagnostic Radiographs**: Standard photographic imaging cannot visualize interproximal decay between tight contacts, subgingival calculus concealed beneath periodontal pockets, or periapical pathology. Dental radiographs remain mandatory.

---

## 4. Detected Pathology Classes

The model detects 5 canonical visual condition categories:

| Index | Canonical Class Name | Clinical Scope |
| :---: | :--- | :--- |
| `0` | `calculus` | Supragingival & visible subgingival dental calculus / tartar encrustations |
| `1` | `caries` | Visible enamel cavitation, dark decay lesions, occlusal pit & fissure caries |
| `2` | `gingivitis` | Marginal gingival erythema, edema, and visible localized gum inflammation |
| `3` | `tooth_discoloration` | Extrinsic stains, intrinsic chromatic variation, enamel fluorosis staining |
| `4` | `oral_ulcer` | Aphthous ulcers, mucosal erosions, and visible canker sores |

---

## 5. Training Datasets & Curation Strategy

### Main Pathology Dataset
- **Name**: `oral-disease.yolov11` (Roboflow Universe: `di-qidb9/oral-disease-tabrb`)
- **Volume**: 10,698 dental photographs
- **Split Distribution**:
  - Train: 8,558 images
  - Validation: 1,070 images
  - Test: 1,070 images
- **Annotations**: 62,720 annotated pathology bounding boxes
- **Control Samples**: 570 empty-label true negative control images (5.33% of dataset)
- **License**: CC BY 4.0

### Auxiliary Hard-Negative Mining Datasets
- **Sources**: `Dental Data Set.yolov11`, `Penyakit Gigi Skripsi.yolov11` (Roboflow Universe)
- **Curation Strategy**: Audited healthy dentition images were mined to identify baseline false positives. Verified healthy negative controls were incorporated to aggressively suppress false-positive discoloration predictions on clean teeth.
- **Leakage Prevention**: Benchmark validation (1,070 images) and test (1,070 images) splits remained strictly untouched throughout all iterations.

---

## 6. Verified Validation Metrics

All metrics represent rigorous evaluation against the **1,070 untouched benchmark validation images**:

| Metric | Measured Value |
|---|---|
| **Precision** | **0.680** |
| **Recall** | **0.666** |
| **mAP@50** | **0.687** |
| **mAP@50-95** | **0.360** |

---

## 7. Production Confidence Thresholds

To balance sensitivity and false-positive suppression across heterogeneous smartphone captures, class-specific cutoffs are enforced:

| Condition Class | Calibrated Threshold | Operational Rationale |
|---|---|---|
| `calculus` | **`0.35`** | High sensitivity for detecting visible supragingival calculus deposits |
| `caries` | **`0.55`** | Conservative cutoff reducing false alerts on occlusal pits and fissures |
| `gingivitis` | **`0.50`** | Balanced threshold for visible gingival erythema |
| `tooth_discoloration` | **`0.65`** | Elevated cutoff preventing natural tooth shade variation from triggering alerts |
| `oral_ulcer` | **`0.65`** | Specific threshold for distinct mucosal ulcerations |
| **Global Fallback** | **`0.50`** | Default cutoff for unmapped or generic queries |

---

## 8. Preprocessing Regression Prevention

During platform testing in DaantShaant, it was found that standard computer-vision contrast enhancements such as **CLAHE (Contrast Limited Adaptive Histogram Equalization)** or repeated lossy JPEG re-encoding introduced chromatic artifacts and suppressed legitimate YOLO detections.

DentalTensor Vision strictly mandates **zero-distortion preprocessing**:
- Original RGB pixels are preserved without adaptive equalization.
- In-memory images are processed without lossy re-encoding.
- A regression test suite validates that raw pixel values remain bit-identical.

---

## 9. Known Limitations & Failure Modes

1. **Subgingival and Interproximal Obscuration**: Cannot detect decay between teeth or tartar deep beneath gums.
2. **Lighting and Glare**: Intense smartphone flash reflections or saliva glints can occasionally cause local false positives or mask small lesions.
3. **Anatomical Mimics**: Deep developmental grooves, racial gingival pigmentation, and dental amalgam margins can visually resemble pathology under poor lighting.
4. **Motion Blur**: Blurry or severely out-of-focus images cause significant confidence drops.

---

## 10. Disclaimer

DentalTensor Vision v1.0 is an artificial intelligence research and visual screening system. It is **not** a certified medical device and is **not** intended to replace professional dental diagnosis, clinical examination, or radiograph evaluation by a licensed dental professional.
