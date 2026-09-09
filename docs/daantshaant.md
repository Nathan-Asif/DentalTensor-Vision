# First Production Integration: DaantShaant

## 1. Provenance & Inception

DentalTensor was conceived, trained, calibrated, and proven while **Nathan Asif** was building and architecting **DaantShaant** for the **Alibaba Cloud Bano Qabil Hackathon 2026**.

DaantShaant is an AI-driven oral health screening and care-navigation platform designed to expand access to early dental screening.

---

## 2. Why DentalTensor Was Separated

During DaantShaant's early architecture phase, it became clear that:
1. **Multimodal LLMs are Ineffective for Low-Level Object Localization**: Relying solely on cloud LLMs (such as GPT-4V, Gemini 1.5 Pro, or Qwen-VL) to detect faint calcifications or millimeter-scale occlusal cavities led to hallucinations, high latency, and high cost per scan.
2. **Dedicated Computer Vision is Mandatory**: A custom-trained intraoral detector provides deterministic, reproducible bounding boxes and confidence scores in tens of milliseconds.
3. **General Community Benefit**: Rather than locking this visual perception capability inside DaantShaant's proprietary full-stack platform, Nathan Asif separated the core computer vision engine into **DentalTensor**—releasing it under the AGPL-3.0 copyleft license for researchers, clinicians, and health-tech developers worldwide.

---

## 3. How DaantShaant Uses DentalTensor

Within DaantShaant, DentalTensor Vision serves as the foundational perception layer:

```
[Patient Dental Photograph]
           │
           ▼
[Semantic Dental Relevance Gate] (Reject non-oral pictures)
           │
           ▼
[Mechanical Image Quality Gate] (Lighting, blur, resolution checks)
           │
           ▼
[DentalTensor Vision v1.0] (Detects caries, calculus, gingivitis, staining, ulcers)
           │
           ▼
[Spatial Aggregation & Evidence Normalization]
           │
           ▼
[Deterministic Clinical Triage Rules] (Urgency classification: low, medium, urgent)
           │
           ▼
[Patient-Friendly Guidance] (Clear next steps and dental navigation)
```
