"""Test image preprocessing and regression protection against CLAHE distortion."""

from __future__ import annotations

import io
import numpy as np
from PIL import Image

from dentaltensor.inference.preprocessing import (
    assert_no_clahe_preprocessing,
    decode_image_to_numpy,
    decode_image_to_pil,
    get_image_bytes,
)


def test_preprocessing_preserves_pixels_without_clahe():
    """REGRESSION TEST: Verify DentalTensor preprocessing preserves raw pixels without CLAHE distortion.

    CLAHE or non-linear contrast adjustments modify pixel values and suppress YOLO detections.
    The decoded image must remain unaltered.
    """
    # Sample realistic intraoral RGB block
    original_pixels = np.array(
        [
            [[100, 120, 140], [105, 125, 145], [110, 130, 150]],
            [[115, 135, 155], [120, 140, 160], [125, 145, 165]],
            [[130, 150, 170], [135, 155, 175], [140, 160, 180]],
        ],
        dtype=np.uint8,
    )

    # 1. Test decode_image_to_numpy from numpy input
    decoded_arr, w, h = decode_image_to_numpy(original_pixels)
    assert w == 3
    assert h == 3
    assert assert_no_clahe_preprocessing(original_pixels, decoded_arr)
    np.testing.assert_array_equal(original_pixels, decoded_arr)

    # 2. Test decode_image_to_pil preserves identical pixel values
    pil_img = decode_image_to_pil(original_pixels)
    np.testing.assert_array_equal(np.array(pil_img), original_pixels)


def test_decode_from_bytes():
    """Verify PNG image bytes decode without distortion."""
    test_arr = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    pil_img = Image.fromarray(test_arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    decoded_arr, w, h = decode_image_to_numpy(raw_bytes)
    assert w == 20
    assert h == 20
    np.testing.assert_array_equal(test_arr, decoded_arr)


def test_get_image_bytes_passthrough():
    """Verify get_image_bytes returns identical raw bytes when bytes are provided."""
    dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRtest_data"
    extracted = get_image_bytes(dummy_bytes)
    assert extracted == dummy_bytes
