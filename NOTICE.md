# Notices and Attributions

## DentalTensor Vision v1.0

DentalTensor Vision is an open-source oral pathology computer-vision model and framework developed by **Nathan Asif**.
First production integration: **DaantShaant**.

Copyright (C) 2026 Nathan Asif.

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0-only).

---

## Third-Party Components and Attributions

### 1. Ultralytics YOLO
- **Architecture**: Ultralytics YOLO11n
- **Project**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **License**: GNU Affero General Public License v3.0 (AGPL-3.0)
- **Notice**: DentalTensor Vision v1.0 is a custom-trained and fine-tuned oral pathology detection model built on the Ultralytics YOLO11n architecture. The underlying YOLO architecture and deep learning framework were created and maintained by Ultralytics Inc. Nathan Asif developed the task-specific dental dataset preparation, fine-tuning, hard-negative mining, confidence calibration, class-specific production thresholds, inference engineering, and standalone SDK packaging.

### 2. Training and Evaluation Datasets
- **Main Pathology Dataset (`oral-disease.yolov11`)**:
  - Source: Roboflow Universe (`di-qidb9/oral-disease-tabrb`)
  - License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Auxiliary Healthy-Control Datasets (`Dental Data Set.yolov11`, `Penyakit Gigi Skripsi.yolov11`)**:
  - Sources: Roboflow Universe (`nathan-asif-blm21/dental-data-set-tta0w`, `nathan-asif-blm21/penyakit-gigi-skripsi-i77mi`)
  - License: CC BY 4.0
  - Role: Hard-negative mining pool for false-positive reduction on clean dentition.
