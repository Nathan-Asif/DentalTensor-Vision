# DentalTensor Roadmap

This document outlines the planned evolutionary stages of **DentalTensor**, preserving the core principle that DentalTensor Vision remains an independent, high-performance visual perception engine.

---

## Current Release: v1.x (Pre-trained Local Inference)

- [x] **Pre-trained Model Weights**: Shipped canonical weights (`models/dentaltensor_vision_v1.0.pt`) ready for immediate inference.
- [x] **Zero Cloud / Account Dependency**: Runs completely offline without API keys, cloud subscriptions, or external accounts.
- [x] **Local CPU & Acceleration**: Standard CPU inference supported by default, with automatic or explicit local acceleration.
- [x] **Calibrated Class Thresholds**: Centralized production cutoffs balancing sensitivity and specificity across 5 oral classes.
- [x] **Zero-Distortion Preprocessing**: Regression protection ensuring no CLAHE or lossy compression passes.
- [x] **Python API & CLI**: Clean developer library (`DentalTensorVision()`) and offline CLI (`dentaltensor`).
- [ ] **v1.1 Patch**: Enhanced client-side batching and async inference pipelines.
- [ ] **v1.2 Patch**: Interactive CLI visualization options for local debug bounding-box previews.
- [ ] **Packaging Improvements**: Wheels and optimized distribution formats.

---

## Next Milestone: v2.0 (Advanced Inference & Deployment Options)

v2.0 will explore high-throughput inference and flexible deployment patterns. Potential areas under active research:

- **Optimized CUDA Workflows**: Dedicated physical GPU optimizations and mixed-precision kernels.
- **High-Throughput Batch Processing**: Vectorized multi-image inference pipelines for high-volume research and clinical datasets.
- **ONNX Export**: Standardized ONNX Runtime deployment for cross-platform desktop and embedded environments.
- **TensorRT Acceleration**: Quantized FP16/INT8 compilation for low-latency clinic gateways.
- **Additional Runtime Backends**: Extended support for alternative deployment runtimes.
- **Optional Hosted Inference**: Community and enterprise serving adapters for multi-tenant microservices.
- **Model Variants**: Specialized lightweight and higher-resolution vision checkpoints.

*(Note: Features listed for v2.0 represent candidate research targets and are subject to engineering validation.)*

---

## Future Milestone: v3.0 (Optional Pluggable AI & Report Layer)

v3.0 will introduce an **optional, pluggable provider layer** for synthesizing patient-friendly explanations and clinician summaries on top of structured DentalTensor findings:

```
                  DentalTensor Vision
                           │
                           ▼
                  structured findings
                           │
                           ▼
                  optional AI provider
                           │
                           ▼
             formatted report / downstream workflow
```

> [!IMPORTANT]
> **Vision Independence Guarantee**: DentalTensor Vision will **always** remain fully functional and independently usable without requiring an LLM or report provider.

### Potential Provider Adapters
- **Qwen**: Local and cloud Qwen-VL / Qwen2.5 series
- **Gemini**: Google Gemini 1.5 / 2.0 Flash
- **OpenAI**: GPT-4o / GPT-4o-mini
- **Claude**: Anthropic Claude 3.5 Sonnet
- **Local Small LLMs**: Ollama / vLLM (Llama-3, Mistral, MedGemma)
- **Custom Provider**: Developer-defined report generation hook

---

## Research Horizons

1. **Instance Segmentation**: Moving from bounding-box detection to polygon segmentation for precise gingival margin mapping and calculus boundary delineation.
2. **Dental Radiography Models**: Training DentalTensor variants specialized on panoramic orthopantomograms (OPGs) and bitewing X-rays for bone loss and interproximal decay.
3. **Expanded Pathology Taxonomy**: Curating fine-grained classes including fluorosis staging, enamel hypoplasia, attrition, and pericoronitis.
4. **Active Learning & Semi-Supervised Mining**: Expanding the automated hard-negative mining pool across diverse ethnic dentition datasets.
