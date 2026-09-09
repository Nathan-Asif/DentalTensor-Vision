# Training Methodology & Configuration

> [!NOTE]
> **Historical Provenance Only**: This document explains how DentalTensor Vision v1.0 was developed and fine-tuned.
> **The model is ALREADY trained.** Users do not need to train the model or download the training dataset.
> Pre-trained weights are distributed as `models/dentaltensor_vision_v1.0.pt` for direct local inference.

---

## 1. Hardware & Environment (Model Development)

- **Architecture**: Ultralytics YOLO11n
- **Hardware Acceleration**: Dedicated GPU runtime (24 GB VRAM)
- **Base Container**: Debian Linux with PyTorch and CUDA 12.x runtimes

---

## 2. Hyperparameters & Settings

```yaml
model: yolo11n.pt
epochs: 12
imgsz: 640
batch: 16
workers: 8
patience: 4
seed: 42
optimizer: AdamW
lr0: 0.0005        # Conservative fine-tuning learning rate
close_mosaic: 3    # Disables mosaic augmentation in final 3 epochs for realistic boundary refinement
box: 7.5           # Bounding-box loss gain
cls: 0.5           # Class classification loss gain
dfl: 1.5           # Distribution focal loss gain
```

---

## 3. Training Dynamics & Convergence

Fine-tuning from the base checkpoint achieved rapid convergence within 12 epochs:
- **Validation Precision**: 0.680
- **Validation Recall**: 0.666
- **mAP@50**: 0.687
- **mAP@50-95**: 0.360

Artifacts produced:
- Canonical Weights: `models/dentaltensor_vision_v1.0.pt`
- Training logs and loss metrics: `assets/metrics/training_results.png`
- Confusion matrix: `assets/metrics/confusion_matrix.png`
