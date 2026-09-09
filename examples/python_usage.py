"""Example script demonstrating DentalTensor Vision Python SDK usage."""

from __future__ import annotations

from pathlib import Path
from dentaltensor import DentalTensorVision


def main():
    print("--- DentalTensor Vision v1.0 Python SDK Example ---")

    # Initialize vision engine with local pre-trained weights
    # Default device is 'auto' (detects compatible GPU acceleration or falls back to CPU)
    vision = DentalTensorVision()
    print(f"Engine: {vision}")
    print("Active calibrated thresholds:")
    for cls, thresh in vision.get_thresholds().items():
        print(f"  {cls}: {thresh}")

    sample_image = "sample_oral_image.jpg"
    if not Path(sample_image).is_file():
        print(f"\nNote: '{sample_image}' not found in current directory.")
        print("To run local inference on your own image, execute:")
        print("    result = vision.predict('path/to/my_image.jpg')")
        return

    print(f"\nRunning local prediction on '{sample_image}'...")
    result = vision.predict(sample_image)

    print("\nStructured Result:")
    print(result.to_json(indent=2))

    print("\nIterating over findings:")
    for finding in result.findings:
        print(
            f"- Found {finding.class_name} "
            f"(confidence: {finding.confidence:.2f}, "
            f"box: {finding.bbox}, "
            f"threshold: {finding.threshold_used})"
        )


if __name__ == "__main__":
    main()
