# Model Architecture & Specifications

## 1. Architectural Foundation

DentalTensor Vision v1.0 is built on the **Ultralytics YOLO11n** nano object detection architecture:

- **Network Layers**: 101 layers
- **Total Parameters**: 2,583,127 parameters
- **Computational Complexity**: 6.4 GFLOPs at 640x640 resolution
- **Inference Latency**: ~15–35 ms on NVIDIA A10G / T4 GPU
- **Input Format**: 3-channel RGB image tensor normalized to [0, 1]

---

## 2. Canonical Classes

DentalTensor Vision maps intraoral visual presentations into 5 distinct categories:

| Index | Label | Scope & Clinical Presentation |
|---|---|---|
| `0` | `calculus` | Supragingival calculus and visible subgingival tartar encrustations along tooth cervical margins. |
| `1` | `caries` | Visible enamel cavitation, dark occlusal pits, and demineralized fissure lesions. |
| `2` | `gingivitis` | Marginal gingival redness (erythema), swelling (edema), and visible inflammation. |
| `3` | `tooth_discoloration` | Extrinsic stains, tobacco/coffee staining, intrinsic fluorosis, and enamel chromatic variation. |
| `4` | `oral_ulcer` | Visible aphthous ulcers, mucosal canker sores, and inflammatory mucosal erosions. |

---

## 3. Discoloration Safety & Separation

In earlier training baselines, generalized yellowing of healthy teeth frequently triggered false cavity or tartar alerts. DentalTensor resolves this through:
- **Isolated Classification**: Tooth discoloration is an independent class and is never merged with caries or calculus.
- **Elevated Production Cutoff (0.65)**: Prevents natural tooth shade variations from triggering alerts.
