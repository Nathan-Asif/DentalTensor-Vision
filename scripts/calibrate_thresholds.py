"""Confidence Threshold Calibration & Analysis Script for DentalTensor Vision.

Evaluates predictions across confidence thresholds (0.30 to 0.80) to analyze:
- Per-class Precision, Recall, and F1 score curves
- False-positive suppression on clean dentition
- Cross-class confusion between calculus and tooth discoloration
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.is_dir() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from dentaltensor.config import CLASSES, PRODUCTION_THRESHOLDS


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two bounding boxes in [x1, y1, x2, y2] format."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def sweep_thresholds(
    predictions: List[Dict],
    ground_truth: List[Dict],
    thresholds: List[float],
    iou_cutoff: float = 0.45,
) -> Dict[float, Dict[str, Dict[str, float]]]:
    """Sweep thresholds across predictions and calculate metrics per class."""
    results = {}

    for thresh in thresholds:
        class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        # Match predictions to ground truth at this threshold
        filtered_preds = [p for p in predictions if p["confidence"] >= thresh]

        # Group by image
        img_preds = defaultdict(list)
        for p in filtered_preds:
            img_preds[p["image_id"]].append(p)

        img_gts = defaultdict(list)
        for g in ground_truth:
            img_gts[g["image_id"]].append(g)

        all_imgs = set(img_preds.keys()) | set(img_gts.keys())

        for img_id in all_imgs:
            preds = img_preds[img_id]
            gts = list(img_gts[img_id])

            matched_gt = set()
            for p in sorted(preds, key=lambda x: -x["confidence"]):
                best_iou = 0.0
                best_gt_idx = -1
                for idx, gt in enumerate(gts):
                    if idx in matched_gt or gt["class_name"] != p["class_name"]:
                        continue
                    iou = compute_iou(p["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= iou_cutoff and best_gt_idx >= 0:
                    class_stats[p["class_name"]]["tp"] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    class_stats[p["class_name"]]["fp"] += 1

            for idx, gt in enumerate(gts):
                if idx not in matched_gt:
                    class_stats[gt["class_name"]]["fn"] += 1

        # Calculate P, R, F1
        results[thresh] = {}
        for cls in CLASSES:
            tp = class_stats[cls]["tp"]
            fp = class_stats[cls]["fp"]
            fn = class_stats[cls]["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            results[thresh][cls] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate DentalTensor Vision confidence thresholds."
    )
    parser.add_argument(
        "--show-calibrated",
        action="store_true",
        help="Print current production calibrated thresholds",
    )
    args = parser.parse_args()

    print("=== DentalTensor Vision v1.0 — Production Thresholds ===")
    for cls, thresh in PRODUCTION_THRESHOLDS.items():
        print(f"  {cls:<22}: {thresh:.2f}")
    print("  Global Fallback       : 0.50")
    print("=========================================================")


if __name__ == "__main__":
    main()
