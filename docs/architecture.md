# Architecture & System Design

DentalTensor Vision is designed as a standalone, lightweight, pre-trained computer-vision perception engine.

---

## 1. High-Level System Flow (v1.0)

```
[Developer App / CLI / API]
           │
           │ (1) Intraoral Image (Path / Bytes / PIL / Numpy)
           ▼
[DentalTensor Vision Python Engine]
   • Zero-Distortion Preprocessing (Raw RGB decode, NO CLAHE, NO lossy compression)
   • Device & Checkpoint Path Resolution
           │
           │ (2) In-Memory RGB Array (640x640)
           ▼
[Local Ultralytics YOLO11n Inference Engine]
   • Pre-Trained Learned Weights (models/dentaltensor_vision_v1.0.pt)
   • Local Compute: Standard CPU (default) or Local Acceleration (CUDA)
   • Forward Pass & Non-Maximum Suppression (IoU: 0.45)
           │
           │ (3) Raw Bounding Boxes & Confidence Scores
           ▼
[Calibration & Formatting Pipeline]
   • Calibrated Confidence Threshold Filtering:
       - calculus: 0.35
       - caries: 0.55
       - gingivitis: 0.50
       - tooth_discoloration: 0.65
       - oral_ulcer: 0.65
       - global fallback: 0.50
   • Pixel & Normalized Coordinate Calculations
           │
           │ (4) Structured Result Payload
           ▼
[DentalVisionResult]
   • List of DentalFinding objects (class_name, confidence, bbox, threshold_used, bbox_normalized)
   • Image metadata (dimensions, channels, format)
   • Latency benchmarking (inference_ms)
```

---

## 2. Compute Topology & Roadmaps

| Version | Compute Topology | Status |
|---|---|---|
| **v1.x** | **Local Pre-Trained Inference** (CPU by default; compatible local acceleration optional) | **Production / Released** |
| **v2.0** | **Advanced Inference & Deployment**: Optimized CUDA workflows, batching, ONNX, TensorRT | Roadmap |
| **v3.0** | **Optional Report Synthesis Layer**: Provider-neutral LLM adapters (Qwen, Gemini, Claude, OpenAI) | Roadmap |

---

## 3. Strict Boundary Rules

1. **Pre-Trained Model Product**: The model is already trained. End users only run inference.
2. **No Cloud Account or API Keys Required**: Executes locally without external dependencies or accounts.
3. **No Training Dataset at Runtime**: The 10,698-image dataset is not required for running inference.
4. **Standalone Operation**: DentalTensor Vision does not depend on Supabase, LangGraph, or cloud LLMs.
5. **Zero Medical Diagnosis Claims**: The model only produces structured visual findings and bounding boxes.
