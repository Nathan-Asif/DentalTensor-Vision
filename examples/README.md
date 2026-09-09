# DentalTensor Vision Examples

This directory contains examples for interacting with DentalTensor Vision v1.0.

DentalTensor Vision v1.0 runs **local inference** directly on pre-trained weights. No cloud account or external API credentials are required.

## 1. Python Usage

See `python_usage.py`:

```python
from dentaltensor import DentalTensorVision

# Initialize vision engine with pre-trained weights
# Automatically uses local compatible accelerator if available, else CPU
vision = DentalTensorVision()

# Predict findings on an intraoral photo
result = vision.predict("my_teeth_photo.jpg")

# Print clean JSON
print(result.to_json(indent=2))

# Access findings programmatically
for finding in result.findings:
    print(finding.class_name, finding.confidence, finding.bbox)
```

## 2. Device Selection

```python
# Force CPU inference
vision_cpu = DentalTensorVision(device="cpu")

# Explicit CUDA acceleration (if supported)
vision_cuda = DentalTensorVision(device="cuda")
```

## 3. Overriding Thresholds

By default, DentalTensor Vision applies class-calibrated production thresholds:
- `calculus`: 0.35
- `caries`: 0.55
- `gingivitis`: 0.50
- `tooth_discoloration`: 0.65
- `oral_ulcer`: 0.65

You can supply a global override when calling `predict`:

```python
# Override all classes with a strict 0.70 confidence threshold
result = vision.predict("my_teeth_photo.jpg", threshold=0.70)
```

## 4. CLI Usage

Run prediction directly from the terminal:

```bash
# Formatted table output
dentaltensor predict path/to/image.jpg

# JSON output
dentaltensor predict path/to/image.jpg --json

# Save JSON to file
dentaltensor predict path/to/image.jpg --output results.json

# Specify compute device
dentaltensor predict path/to/image.jpg --device cpu

# Override threshold
dentaltensor predict path/to/image.jpg --threshold 0.60
```
