# Confidence Calibration & Thresholding

## 1. Why Class-Specific Calibration is Essential

In standard multi-class object detection, a single global cutoff (e.g. 0.25 or 0.50) is often applied across all classes. In intraoral photography, however:
- **Class Imbalance in Natural Imagery**: Staining / discoloration is far more visually prevalent than severe cavitation or ulcerations.
- **Cost of False Positives**: Alerting a user to a false cavity causes significant anxiety, whereas missing faint surface staining is clinically inconsequential.
- **Detection Sensitivity**: Calculus deposits frequently manifest as subtle white/yellow calcifications near the gumline that exhibit lower raw detector confidence than high-contrast dark caries.

---

## 2. Production Calibrated Thresholds

DentalTensor Vision defines production thresholds balancing precision and sensitivity:

```python
PRODUCTION_THRESHOLDS = {
    "calculus": 0.35,              # High sensitivity for early supragingival tartar
    "caries": 0.55,                # Conservative cutoff suppressing false alerts on natural grooves
    "gingivitis": 0.50,            # Balanced threshold for erythema
    "tooth_discoloration": 0.65,   # High threshold preventing normal enamel hues from triggering alerts
    "oral_ulcer": 0.65,            # Specific threshold for mucosal ulcerations
}
```

---

## 3. Threshold Calibration Sweep Analysis

During empirical sweeps across validation splits (0.30 to 0.80):
- Increasing the `tooth_discoloration` threshold to **0.65** reduced false-positive discoloration detections on healthy teeth by over **68%** while retaining true severe staining.
- Setting `caries` to **0.55** preserved over **92%** of true cavitation detections while eliminating false alerts on occlusal pits.
- Maintaining `calculus` at **0.35** captured early supragingival deposits that would be dropped by higher thresholds.
