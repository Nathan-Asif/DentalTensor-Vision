"""Evaluation script for DentalTensor Vision models on YOLO-format datasets.

Computes Precision, Recall, mAP@50, mAP@50-95 across test/validation benchmarks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.is_dir() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from dentaltensor.config import BENCHMARK_METRICS, FULL_MODEL_NAME


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DentalTensor Vision on benchmark split."
    )
    parser.add_argument(
        "--model",
        "-m",
        default="models/dentaltensor_vision_v1.0.pt",
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data",
        "-d",
        required=False,
        help="Path to data.yaml dataset descriptor",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--show-verified-benchmark",
        action="store_true",
        help="Print verified reference benchmark metrics",
    )
    args = parser.parse_args()

    if args.show_verified_benchmark or not args.data:
        print(f"=== {FULL_MODEL_NAME} — Verified Reference Benchmark ===")
        print(f"Training Dataset Size : {int(BENCHMARK_METRICS['training_images'])} images")
        print(f"Precision             : {BENCHMARK_METRICS['precision']:.3f}")
        print(f"Recall                : {BENCHMARK_METRICS['recall']:.3f}")
        print(f"mAP@50                : {BENCHMARK_METRICS['map50']:.3f}")
        print(f"mAP@50-95             : {BENCHMARK_METRICS['map50_95']:.3f}")
        if not args.data:
            print("\nTo evaluate against a live dataset, provide '--data path/to/data.yaml'")
            return

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Error: 'ultralytics' is required for local evaluation. "
            "Install with: pip install ultralytics",
            file=sys.stderr,
        )
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Error: Model checkpoint not found at '{model_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {model_path}")
    model = YOLO(str(model_path))

    print(f"Evaluating on '{args.data}' (split: {args.split})...")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=640,
        batch=16,
        verbose=True,
    )

    print("\n--- Evaluation Results ---")
    print(f"Precision : {metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"Recall    : {metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
    print(f"mAP@50    : {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP@50-95 : {metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


if __name__ == "__main__":
    main()
