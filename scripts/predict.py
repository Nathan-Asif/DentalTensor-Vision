"""Standalone prediction script for DentalTensor Vision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path if running from repository clone
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.is_dir() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from dentaltensor.vision import DentalTensorVision


def main():
    parser = argparse.ArgumentParser(
        description="Run local DentalTensor Vision prediction on an image."
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--output", "-o", help="Path to save output JSON")
    parser.add_argument(
        "--threshold", "-t", type=float, default=None, help="Global threshold override"
    )
    parser.add_argument(
        "--device", "-d", type=str, default="auto", help="Inference device ('auto', 'cpu', 'cuda')"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Optional path to model checkpoint (.pt)"
    )
    args = parser.parse_args()

    vision = DentalTensorVision(model_path=args.model, device=args.device)
    result = vision.predict(args.image, threshold=args.threshold)

    if args.output:
        Path(args.output).write_text(result.to_json(indent=2), encoding="utf-8")

    if args.json:
        print(result.to_json(indent=2))
    else:
        print(f"Model: {result.model} v{result.version}")
        print(f"Device: {vision.device}")
        print(f"Findings: {len(result.findings)}")
        for f in result.findings:
            print(f"  [{f.class_name}] conf={f.confidence:.2f} bbox={f.bbox}")


if __name__ == "__main__":
    main()
