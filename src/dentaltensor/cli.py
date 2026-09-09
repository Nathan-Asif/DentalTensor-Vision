"""Command-line interface (CLI) for DentalTensor Vision."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from dentaltensor.config import (
    BENCHMARK_METRICS,
    CANONICAL_CHECKPOINT_NAME,
    CLASSES,
    DEVELOPER,
    FIRST_INTEGRATION,
    FULL_MODEL_NAME,
    PRODUCTION_THRESHOLDS,
    VERSION,
)
from dentaltensor.schemas import DentalVisionResult
from dentaltensor.vision import DentalTensorVision


def print_banner() -> None:
    """Print clean DentalTensor CLI banner."""
    print("=" * 60)
    print(f"  {FULL_MODEL_NAME}")
    print(f"  Developer: {DEVELOPER}")
    print(f"  First Production Integration: {FIRST_INTEGRATION}")
    print("=" * 60)


def cmd_version(args: argparse.Namespace) -> int:
    """Print version details."""
    print(f"dentaltensor-vision {VERSION}")
    print(f"Model: {FULL_MODEL_NAME}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Print comprehensive model, threshold, and inference info."""
    print_banner()
    print("\nModel Architecture & Foundation:")
    print("  Base Architecture : Ultralytics YOLO11n")
    print("  Training Dataset  : 10,698 dental images (pre-training completed)")
    print("  Validation Set    : 1,070 untouched benchmark images")
    print("  Test Set          : 1,070 untouched benchmark images")
    print(f"  Model Weights     : models/{CANONICAL_CHECKPOINT_NAME}")
    print("\nVerified Engineering Metrics:")
    print(f"  Precision         : {BENCHMARK_METRICS['precision']:.3f}")
    print(f"  Recall            : {BENCHMARK_METRICS['recall']:.3f}")
    print(f"  mAP@50            : {BENCHMARK_METRICS['map50']:.3f}")
    print(f"  mAP@50-95         : {BENCHMARK_METRICS['map50_95']:.3f}")
    print("\nCalibrated Class Thresholds:")
    for cls, thresh in PRODUCTION_THRESHOLDS.items():
        print(f"  - {cls:<20}: {thresh:.2f}")
    print(f"  - Global Fallback     : 0.50")
    print("\nInference Mode:")
    print("  Local Pre-Trained Inference (CPU / compatible local acceleration)")
    print("  Zero cloud setup, zero API keys, and no training dataset required.")
    print("=" * 60)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Execute prediction on an image file."""
    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Error: Image file not found at '{image_path}'", file=sys.stderr)
        return 1

    try:
        vision = DentalTensorVision(
            model_path=args.model,
            device=args.device,
        )
        result: DentalVisionResult = vision.predict(
            str(image_path),
            threshold=args.threshold,
            device=args.device,
            source=image_path.name,
        )
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 2

    # If output path is requested, write to file
    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(result.to_json(indent=2), encoding="utf-8")
        if not args.json:
            print(f"Saved results to: {out_p}")

    # Display output
    if args.json:
        print(result.to_json(indent=2))
        return 0

    # Human-readable formatted console output
    print_banner()
    print(f"\nImage: {image_path.resolve()}")
    if result.image_metadata:
        m = result.image_metadata
        print(f"Resolution: {m.width}x{m.height} ({m.format})")
    if result.inference_ms is not None:
        print(f"Execution Latency: {result.inference_ms} ms (Device: {vision.device})")

    if args.threshold is not None:
        print(f"Threshold Override: {args.threshold:.2f}")
    else:
        print("Thresholds: Calibrated per-class production cutoffs")

    print("\nFindings:")
    if not result.findings:
        print("  No visible oral pathologies detected above confidence thresholds.")
    else:
        print(f"  Total findings: {len(result.findings)}\n")
        print(f"  {'Class':<22} {'Confidence':<12} {'Threshold':<12} {'Bounding Box [x1, y1, x2, y2]'}")
        print("  " + "-" * 70)
        for f in result.findings:
            bbox_str = f"[{f.bbox[0]:.1f}, {f.bbox[1]:.1f}, {f.bbox[2]:.1f}, {f.bbox[3]:.1f}]"
            print(f"  {f.class_name:<22} {f.confidence*100:>5.1f}%      {f.threshold_used:>5.2f}        {bbox_str}")

    print("=" * 60)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        prog="dentaltensor",
        description=(
            "DentalTensor Vision v1.0 — Pre-trained oral pathology computer vision "
            "built on Ultralytics YOLO11n.\n"
            "Developed by Nathan Asif. First production integration: DaantShaant."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run oral pathology detection on an intraoral image",
        description="Run local DentalTensor Vision inference on an image file.",
    )
    predict_parser.add_argument(
        "image",
        type=str,
        help="Path to input oral image (JPEG, PNG, WebP)",
    )
    predict_parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw structured JSON findings to stdout",
    )
    predict_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save structured JSON findings to specified file path",
    )
    predict_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help=(
            "Global confidence threshold override (0.0 - 1.0). "
            "If omitted, calibrated class-specific thresholds are used "
            "(calculus: 0.35, caries: 0.55, gingivitis: 0.50, "
            "tooth_discoloration: 0.65, oral_ulcer: 0.65)."
        ),
    )
    predict_parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        help="Inference device: 'auto' (default), 'cpu', or 'cuda'",
    )
    predict_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.pt file)",
    )

    # Info subcommand
    subparsers.add_parser(
        "info",
        help="Show model architecture, metrics, and threshold details",
    )

    # Version subcommand
    subparsers.add_parser(
        "version",
        help="Show version information",
    )

    return parser


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 0

    args = parser.parse_args()

    if args.command == "predict":
        return cmd_predict(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "version":
        return cmd_version(args)
    else:
        parser.print_help(sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
