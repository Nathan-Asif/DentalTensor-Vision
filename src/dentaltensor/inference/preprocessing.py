"""Image decoding and preprocessing for DentalTensor Vision.

CRITICAL INFERENCE REGRESSION RULE:
During DaantShaant development, it was discovered that applying contrast enhancements
such as CLAHE (Contrast Limited Adaptive Histogram Equalization) or aggressive lossy
JPEG re-encoding distorts subtle intraoral pathology cues and suppresses real YOLO detections.
DentalTensor Vision requires the decoded original image without unnecessary destructive preprocessing.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
from PIL import Image


ImageInputType = Union[str, Path, bytes, bytearray, Image.Image, np.ndarray]


class ImagePreprocessingError(ValueError):
    """Raised when an image cannot be read, decoded, or validated."""
    pass


MAX_EDGE_PX: int = 1024


def decode_image_to_pil(image_input: ImageInputType) -> Image.Image:
    """Decode any supported image input into an RGB PIL Image without destructive preprocessing.

    Preserves original pixel values without applying CLAHE or lossy compression passes.
    """
    if isinstance(image_input, Image.Image):
        if image_input.mode != "RGB":
            return image_input.convert("RGB")
        return image_input

    if isinstance(image_input, np.ndarray):
        # Support both HxW (grayscale), HxWxC (RGB / BGR)
        if image_input.ndim == 2:
            return Image.fromarray(image_input).convert("RGB")
        elif image_input.ndim == 3:
            # Assume RGB unless specified otherwise
            return Image.fromarray(image_input.astype(np.uint8))
        raise ImagePreprocessingError(
            f"Invalid numpy array shape {image_input.shape}. Expected 2D or 3D array."
        )

    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if path.is_file():
            try:
                with open(path, "rb") as f:
                    data = f.read()
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as exc:
                raise ImagePreprocessingError(
                    f"Failed to read image file at '{path}': {exc}"
                ) from exc

        # If it's a string, it might be base64-encoded
        if isinstance(image_input, str):
            raw_str = image_input.strip()
            # Strip data URL prefix if present
            if raw_str.startswith("data:image"):
                raw_str = raw_str.split(",", 1)[-1]
            try:
                decoded_bytes = base64.b64decode(raw_str)
                return Image.open(io.BytesIO(decoded_bytes)).convert("RGB")
            except Exception:
                pass

        raise ImagePreprocessingError(
            f"Image path does not exist and is not valid base64: '{image_input}'"
        )

    if isinstance(image_input, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        except Exception as exc:
            raise ImagePreprocessingError(
                f"Failed to decode image bytes: {exc}"
            ) from exc

    raise ImagePreprocessingError(
        f"Unsupported image input type: {type(image_input).__name__}"
    )


def decode_image_bytes_to_bgr(raw: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR NumPy array.

    Uses OpenCV imdecode first; if unavailable or decoding returns None,
    falls back to PIL and converts to BGR via channel inversion.
    """
    try:
        import cv2

        arr = np.frombuffer(raw, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is not None:
            return decoded
    except Exception:
        pass

    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(pil)[:, :, ::-1]
    except Exception as exc:
        raise ImagePreprocessingError(
            f"Failed to decode image bytes to BGR: {exc}"
        ) from exc


def decode_image_to_bgr(
    image_input: ImageInputType,
) -> Tuple[np.ndarray, int, int, Optional[str]]:
    """Decode image input into a BGR NumPy array matching DaantShaant production semantics.

    Returns:
        (image_bgr, width, height, source_filename)
        source_filename is the file's basename if input was a path, else None.
    """
    source_name: Optional[str] = None

    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if path.is_file():
            source_name = path.name
            try:
                raw_bytes = path.read_bytes()
            except Exception as exc:
                raise ImagePreprocessingError(
                    f"Failed to read image file at '{path}': {exc}"
                ) from exc
            image_bgr = decode_image_bytes_to_bgr(raw_bytes)
            height, width = image_bgr.shape[:2]
            return image_bgr, width, height, source_name

        if isinstance(image_input, str):
            raw_str = image_input.strip()
            if raw_str.startswith("data:image"):
                raw_str = raw_str.split(",", 1)[-1]
            try:
                decoded_bytes = base64.b64decode(raw_str)
                image_bgr = decode_image_bytes_to_bgr(decoded_bytes)
                height, width = image_bgr.shape[:2]
                return image_bgr, width, height, None
            except Exception:
                pass

        raise ImagePreprocessingError(
            f"Image path does not exist and is not valid base64: '{image_input}'"
        )

    if isinstance(image_input, (bytes, bytearray)):
        image_bgr = decode_image_bytes_to_bgr(bytes(image_input))
        height, width = image_bgr.shape[:2]
        return image_bgr, width, height, None

    if isinstance(image_input, Image.Image):
        pil = image_input.convert("RGB")
        image_bgr = np.array(pil)[:, :, ::-1]
        height, width = image_bgr.shape[:2]
        return image_bgr, width, height, None

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            try:
                import cv2

                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            except Exception:
                image_bgr = np.stack([image_input] * 3, axis=-1)
        elif image_input.ndim == 3:
            image_bgr = image_input.astype(np.uint8)
        else:
            raise ImagePreprocessingError(
                f"Invalid numpy array shape {image_input.shape}. Expected 2D or 3D array."
            )
        height, width = image_bgr.shape[:2]
        return image_bgr, width, height, None

    raise ImagePreprocessingError(
        f"Unsupported image input type: {type(image_input).__name__}"
    )


def normalize_image(image: np.ndarray, max_edge: int = MAX_EDGE_PX) -> np.ndarray:
    """Normalize image by resizing only when exceeding max_edge, preserving smaller images.

    Ports the canonical DaantShaant normalize_image() semantics.
    Uses cv2.INTER_AREA interpolation for area-based downsampling.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_edge:
        return image

    try:
        import cv2

        scale = max_edge / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        pil_img = Image.fromarray(image[:, :, ::-1])
        pil_img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        return np.array(pil_img)[:, :, ::-1]


def decode_image_to_numpy(image_input: ImageInputType) -> Tuple[np.ndarray, int, int]:
    """Decode image input to an RGB numpy array and return (array, width, height).

    Guarantees no CLAHE or unneeded re-compression is applied.
    """
    pil_img = decode_image_to_pil(image_input)
    arr = np.array(pil_img)
    height, width = arr.shape[:2]
    return arr, width, height


def get_image_bytes(image_input: ImageInputType) -> bytes:
    """Extract raw image bytes without re-encoding when original bytes/file are available.

    If re-encoding is necessary (e.g. from an in-memory numpy array or raw PIL image),
    uses high-quality PNG or maximum quality JPEG to avoid compression artifacts.
    """
    if isinstance(image_input, (bytes, bytearray)):
        return bytes(image_input)

    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if path.is_file():
            return path.read_bytes()

    # For PIL / numpy, convert to lossless PNG bytes to guarantee no compression artifacts
    pil_img = decode_image_to_pil(image_input)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def assert_no_clahe_preprocessing(
    input_array: np.ndarray, processed_array: np.ndarray
) -> bool:
    """Regression test assertion helper: verifies that image values were not altered by CLAHE.

    In the presence of CLAHE, pixel intensities undergo non-linear histogram equalization,
    causing input_array and processed_array to diverge even at identical dimensions.
    """
    if input_array.shape != processed_array.shape:
        return False
    return np.array_equal(input_array, processed_array)
