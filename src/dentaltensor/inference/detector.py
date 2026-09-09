"""Core Ultralytics YOLO11n oral pathology detector for DentalTensor Vision."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dentaltensor.config import (
    CLASSES,
    FULL_MODEL_NAME,
    PRODUCTION_THRESHOLDS,
    default_config,
    resolve_device,
    resolve_model_path,
)
from dentaltensor.inference.postprocessing import (
    build_vision_result,
    filter_and_format_findings,
)
from dentaltensor.inference.preprocessing import (
    ImageInputType,
    decode_image_to_bgr,
    normalize_image,
)
from dentaltensor.inference.thresholds import (
    YOLO_INDEX_TO_CLASS,
    get_confidence_threshold,
    normalize_class_name,
)
from dentaltensor.schemas import DentalVisionResult


class DentalTensorInferenceError(RuntimeError):
    """Raised when inference crashes, weights cannot be loaded, or dependencies are missing."""
    pass


class DentalTensorDetector:
    """Ultralytics YOLO11n wrapper for DentalTensor Vision v1.0 local inference."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        device: Optional[str] = "auto",
    ):
        self.model_path: Path = resolve_model_path(model_path)
        self.device: str = resolve_device(device)
        self._model: Any = None

    def load_model(self) -> Any:
        """Lazy-load the Ultralytics YOLO model from pre-trained weights."""
        if self._model is not None:
            return self._model

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"DentalTensor Vision model checkpoint not found at '{self.model_path}'. "
                f"DentalTensor Vision v1.0 requires the pre-trained weights file 'models/dentaltensor_vision_v1.0.pt'."
            )

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
            return self._model
        except ImportError as exc:
            raise DentalTensorInferenceError(
                "Ultralytics is required for DentalTensor Vision local inference: "
                "pip install ultralytics"
            ) from exc
        except Exception as exc:
            raise DentalTensorInferenceError(
                f"Failed to load DentalTensor weights from '{self.model_path}': {exc}"
            ) from exc

    def detect(
        self,
        image_input: ImageInputType,
        threshold_override: Optional[float] = None,
        source_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> DentalVisionResult:
        """Execute forward pass on image and return structured DentalVisionResult."""
        image_bgr, orig_w, orig_h, detected_source = decode_image_to_bgr(image_input)
        resolved_source = source_id or detected_source

        # Port canonical normalize_image semantics (resizes only if > 1024 max edge)
        image_bgr = normalize_image(image_bgr)
        h, w = image_bgr.shape[:2]

        model = self.load_model()
        active_device = resolve_device(device) if device is not None else self.device

        # Candidate YOLO confidence: minimum of class thresholds or explicit override
        if threshold_override is not None:
            min_predict_conf = float(threshold_override)
        else:
            min_predict_conf = min(PRODUCTION_THRESHOLDS.values())

        start_time = time.perf_counter()

        try:
            results = model.predict(
                source=image_bgr,
                conf=min_predict_conf,
                iou=0.45,
                verbose=False,
                device=active_device,
            )
        except Exception as exc:
            raise DentalTensorInferenceError(f"YOLO forward pass failed on {active_device}: {exc}") from exc

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        raw_detections: List[Dict[str, Any]] = []
        if results and len(results) > 0:
            res = results[0]
            boxes = getattr(res, "boxes", None)
            names = getattr(res, "names", {})

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()

                for i in range(len(cls)):
                    c_idx = int(cls[i])
                    raw_name = names.get(c_idx, YOLO_INDEX_TO_CLASS.get(c_idx, str(c_idx)))
                    raw_detections.append(
                        {
                            "class_name": raw_name,
                            "confidence": float(conf[i]),
                            "bbox": [
                                float(xyxy[i][0]),
                                float(xyxy[i][1]),
                                float(xyxy[i][2]),
                                float(xyxy[i][3]),
                            ],
                        }
                    )

        findings = filter_and_format_findings(
            raw_detections=raw_detections,
            image_width=w,
            image_height=h,
            threshold_override=threshold_override,
        )

        return build_vision_result(
            findings=findings,
            image_width=w,
            image_height=h,
            inference_ms=elapsed_ms,
            source_identifier=resolved_source,
        )
